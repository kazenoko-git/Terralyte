# imagen.py (FULLY PATCHED & OPTIMIZED)
import os, math, mercantile, asyncio, aiofiles, httpx
from pathlib import Path
from io import BytesIO
from PIL import Image
from typing import Optional

# -------------------------------------------------------------------
# Google, Bing tile URLs
# -------------------------------------------------------------------
def tile_xy_to_quadkey(x: int, y: int, z: int) -> str:
    quadkey = []
    for i in range(z, 0, -1):
        bit = 0
        mask = 1 << (i - 1)
        if x & mask:
            bit += 1
        if y & mask:
            bit += 2
        quadkey.append(str(bit))
    return "".join(quadkey)


# -------------------------------------------------------------------
# Clean Mercator Pixel Mapping — SAME as Google Maps
# -------------------------------------------------------------------
def latlon_to_pixel_in_stitched(lat, lon, zoom, radius, tile_size=256):
    """
    Correct global_webmercator_pixel → stitched_local_pixel mapping.
    """

    # ---- 1. GLOBAL pixel coords (Google WebMercator) ----
    sin_lat = math.sin(math.radians(lat))
    n = 2.0 ** zoom

    x_global = ((lon + 180.0) / 360.0) * n * tile_size
    y_global = ((1 - math.log((1 + sin_lat) / (1 - sin_lat)) / math.pi) / 2) * n * tile_size

    # ---- 2. Top-left tile in stitched grid ----
    center_tile = mercantile.tile(lon, lat, zoom)
    top_left_tile_x = center_tile.x - radius
    top_left_tile_y = center_tile.y - radius

    # ---- 3. Local stitched image coords ----
    px = x_global - (top_left_tile_x * tile_size)
    py = y_global - (top_left_tile_y * tile_size)

    return px, py


# Simple center crop for fallback
def crop_center(img, crop_width=640, crop_height=640):
    w, h = img.size
    left = int((w - crop_width) / 2)
    top = int((h - crop_height) / 2)
    return img.crop((left, top, left + crop_width, top + crop_height))


# -------------------------------------------------------------------
# IMAGEN CLASS — optimized, async tile fetch, caching, 3 providers
# -------------------------------------------------------------------
class Imagen:
    def __init__(
        self,
        provider: str = "esri",
        cache_dir: str = "./tile_cache",
        timeout: int = 12,
        concurrency: int = 15,
        user_agent: str = "TerralyteTileFetcher/1.0",
    ):
        self.provider = provider.lower()

        if self.provider not in ("esri", "google", "bing"):
            raise ValueError("Provider must be one of: esri, google, bing")

        # Keys passed from Tauri (environment)
        self.GMAPS_KEY = os.environ.get("GMAPS_KEY", "")
        self.BING_KEY = os.environ.get("BING_KEY", "")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = timeout
        self.concurrency = concurrency
        self.headers = {"User-Agent": user_agent}

    # ---------------------- URL BUILDERS ----------------------
    def _esri_url(self, x, y, z):
        return f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    def _google_url(self, x, y, z):
        if not self.GMAPS_KEY:
            raise RuntimeError("Google tile key missing (GMAPS_KEY).")
        return f"https://mts0.google.com/vt?lyrs=s&x={x}&y={y}&z={z}&key={self.GMAPS_KEY}"

    def _bing_url(self, x, y, z):
        if not self.BING_KEY:
            raise RuntimeError("Bing tile key missing (BING_KEY).")
        qk = tile_xy_to_quadkey(x, y, z)
        return f"https://t.ssl.ak.tiles.virtualearth.net/tiles/a{qk}.jpeg?g=131&key={self.BING_KEY}"

    # ---------------------- PROVIDER CHOICE ----------------------
    def _tile_url(self, x, y, z):
        if self.provider == "esri":
            return self._esri_url(x, y, z)
        if self.provider == "google":
            return self._google_url(x, y, z)
        if self.provider == "bing":
            return self._bing_url(x, y, z)
        raise RuntimeError("Unknown provider.")

    # ---------------------- SYNC FALLBACK FETCH ----------------------
    def _down_sync(self, x, y, z):
        url = self._tile_url(x, y, z)
        import requests
        r = requests.get(url, timeout=self.timeout, headers=self.headers)
        if r.status_code != 200:
            raise RuntimeError(f"Tile download failed [{r.status_code}] {url}")
        return Image.open(BytesIO(r.content)).convert("RGB")

    # ---------------------- ASYNC FETCH ----------------------
    async def _fetch_tile(self, client, x, y, z):
        cache_path = self.cache_dir / f"{self.provider}_{z}_{x}_{y}.jpg"

        if cache_path.exists():
            async with aiofiles.open(cache_path, "rb") as f:
                data = await f.read()
            return Image.open(BytesIO(data)).convert("RGB")

        url = self._tile_url(x, y, z)

        try:
            r = await client.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.content

            async with aiofiles.open(cache_path, "wb") as f:
                await f.write(data)

            return Image.open(BytesIO(data)).convert("RGB")

        except Exception:
            # fallback to sync
            return self._down_sync(x, y, z)

    # ---------------------- STITCHED TILE GRID ----------------------
    async def _fetch_all_tiles(self, lat, lon, zoom, radius):
        center = mercantile.tile(lon, lat, zoom)

        async with httpx.AsyncClient(follow_redirects=True, headers=self.headers) as client:
            tasks = []

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    tx = center.x + dx
                    ty = center.y + dy
                    tasks.append((dx, dy, asyncio.create_task(self._fetch_tile(client, tx, ty, zoom))))

            # Wait for all tiles
            results = {}
            for dx, dy, t in tasks:
                results[(dx, dy)] = await t

        return results

    # ---------------------- MAIN API ----------------------
    def getStitchedTiles(self, lat, lon, zoom, radius=1):
        """
        Assembles (2r+1)x(2r+1) tile grid centered on (lat,lon).
        """

        # ---- run async fetcher ----
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tile_map = loop.run_until_complete(self._fetch_all_tiles(lat, lon, zoom, radius))

        # ---- Stitch tiles in correct order ----
        rows = []
        for dy in range(-radius, radius + 1):
            row_tiles = []
            for dx in range(-radius, radius + 1):
                row_tiles.append(tile_map[(dx, dy)])
            rows.append(self._hstack(row_tiles))

        final = self._vstack(rows)
        return final

    # ---------------------- STACKERS ----------------------
    def _hstack(self, images):
        widths, heights = zip(*[img.size for img in images])
        out = Image.new("RGB", (sum(widths), max(heights)))
        x = 0
        for img in images:
            out.paste(img, (x, 0))
            x += img.width
        return out

    def _vstack(self, images):
        widths, heights = zip(*[img.size for img in images])
        out = Image.new("RGB", (max(widths), sum(heights)))
        y = 0
        for img in images:
            out.paste(img, (0, y))
            y += img.height
        return out
