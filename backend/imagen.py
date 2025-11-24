# imagen.py (patched: clamp indices, fill-missing tiles, robust stitching)
import os, math, mercantile, asyncio, aiofiles, httpx
from pathlib import Path
from io import BytesIO
from PIL import Image
from typing import Optional

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

def latlon_to_pixel_in_stitched(lat, lon, zoom, radius, tile_size=256):
    """
    Global pixel -> local stitched pixel mapping (web mercator).
    Returns (px, py) local to stitched image top-left.
    """
    sin_lat = math.sin(math.radians(lat))
    n = 2.0 ** zoom

    x_global = ((lon + 180.0) / 360.0) * n * tile_size
    y_global = ((1 - math.log((1 + sin_lat) / (1 - sin_lat)) / math.pi) / 2) * n * tile_size

    center_tile = mercantile.tile(lon, lat, zoom)
    top_left_tile_x = center_tile.x - radius
    top_left_tile_y = center_tile.y - radius

    px = x_global - (top_left_tile_x * tile_size)
    py = y_global - (top_left_tile_y * tile_size)

    return px, py

def crop_center(img, crop_width=640, crop_height=640):
    w, h = img.size
    left = int((w - crop_width) / 2)
    top = int((h - crop_height) / 2)
    return img.crop((left, top, left + crop_width, top + crop_height))

class Imagen:
    def __init__(
        self,
        provider: str = "esri",
        cache_dir: str = "./tile_cache",
        timeout: int = 12,
        concurrency: int = 12,
        user_agent: str = "TerralyteTileFetcher/1.0",
    ):
        self.provider = provider.lower()
        if self.provider not in ("esri", "google", "bing"):
            raise ValueError("Provider must be one of: esri, google, bing")

        self.GMAPS_KEY = os.environ.get("GMAPS_KEY", "") or os.environ.get("VITE_GOOGLE_MAPS_API_KEY", "")
        self.BING_KEY = os.environ.get("BING_KEY", "")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = timeout
        self.concurrency = concurrency
        self.headers = {"User-Agent": user_agent}

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

    def _tile_url(self, x, y, z):
        if self.provider == "esri":
            return self._esri_url(x, y, z)
        if self.provider == "google":
            return self._google_url(x, y, z)
        if self.provider == "bing":
            return self._bing_url(x, y, z)
        raise RuntimeError("Unknown provider.")

    def _down_sync(self, x, y, z):
        url = self._tile_url(x, y, z)
        import requests
        r = requests.get(url, timeout=self.timeout, headers=self.headers)
        if r.status_code != 200:
            raise RuntimeError(f"Tile download failed [{r.status_code}] {url}")
        return Image.open(BytesIO(r.content)).convert("RGB")

    async def _fetch_tile(self, client, x, y, z):
        # clamp tile indices to valid range
        max_tile = (2 ** z) - 1
        tx = max(0, min(max_tile, int(x)))
        ty = max(0, min(max_tile, int(y)))

        cache_path = self.cache_dir / f"{self.provider}_{z}_{tx}_{ty}.jpg"
        if cache_path.exists():
            import aiofiles
            async with aiofiles.open(cache_path, "rb") as f:
                data = await f.read()
            return Image.open(BytesIO(data)).convert("RGB")

        url = self._tile_url(tx, ty, z)
        try:
            r = await client.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.content
            import aiofiles
            async with aiofiles.open(cache_path, "wb") as f:
                await f.write(data)
            return Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            # fallback to sync (robust)
            return self._down_sync(tx, ty, z)

    async def _fetch_all_tiles(self, lat, lon, zoom, radius):
        center = mercantile.tile(lon, lat, zoom)
        # ensure center.x/y are ints
        cx, cy = int(center.x), int(center.y)
        tiles = {}
        async with httpx.AsyncClient(follow_redirects=True, headers=self.headers) as client:
            tasks = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    tx = cx + dx
                    ty = cy + dy
                    # clamp before scheduling so we don't create crazy out-of-range requests
                    max_tile = (2 ** zoom) - 1
                    txc = max(0, min(max_tile, tx))
                    tyc = max(0, min(max_tile, ty))
                    tasks.append((dx, dy, asyncio.create_task(self._fetch_tile(client, txc, tyc, zoom))))

            # gather concurrently & populate map
            for dx, dy, t in tasks:
                try:
                    tiles[(dx, dy)] = await t
                except Exception:
                    # if a tile fails, use a blank tile (same size as typical tile)
                    tiles[(dx, dy)] = Image.new("RGB", (256, 256), (0, 0, 0))
        return tiles

    def getStitchedTiles(self, lat, lon, zoom, radius=0):
        """
        Assemble (2r+1)x(2r+1) grid; missing tiles are filled with blank tiles so the final geometry is stable.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tile_map = loop.run_until_complete(self._fetch_all_tiles(lat, lon, zoom, radius))

        # find tile size from any tile (fallback to 256)
        sample = next(iter(tile_map.values()))
        tile_w, tile_h = sample.size if sample is not None else (256, 256)

        rows = []
        for dy in range(-radius, radius + 1):
            row_tiles = []
            for dx in range(-radius, radius + 1):
                t = tile_map.get((dx, dy))
                if t is None:
                    t = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
                # ensure exact tile size
                if t.size != (tile_w, tile_h):
                    t = t.resize((tile_w, tile_h))
                row_tiles.append(t)
            # horizontal concat
            total_w = sum(t.width for t in row_tiles)
            max_h = max(t.height for t in row_tiles)
            row_img = Image.new("RGB", (total_w, max_h))
            x = 0
            for t in row_tiles:
                row_img.paste(t, (x, 0))
                x += t.width
            rows.append(row_img)

        # vertical concat
        if not rows:
            return Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
        final_w = max(r.width for r in rows)
        final_h = sum(r.height for r in rows)
        final = Image.new("RGB", (final_w, final_h))
        y = 0
        for r in rows:
            final.paste(r, (0, y))
            y += r.height

        return final
