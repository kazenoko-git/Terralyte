# imagen.py (updated)
import asyncio, mercantile, httpx, aiofiles, requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Optional
from tqdm.asyncio import tqdm_asyncio
import math

# helper quadkey for bing
def tile_xy_to_quadkey(x: int, y: int, z: int) -> str:
    quadkey = []
    for i in range(z, 0, -1):
        bit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            bit += 1
        if (y & mask) != 0:
            bit += 2
        quadkey.append(str(bit))
    return "".join(quadkey)

# simple center crop (kept for backward compat)
def crop_center(img, crop_width=640, crop_height=640):
    w, h = img.size
    left = int((w - crop_width) / 2)
    top = int((h - crop_height) / 2)
    right = left + crop_width
    bottom = top + crop_height
    return img.crop((left, top, right, bottom))

def latlon_to_pixel_in_stitched(lat, lon, zoom, radius, tile_size=256):
    """
    Compute pixel coordinates of (lat,lon) inside a stitched grid produced
    by getStitchedTiles(..., radius).
    Returns (px, py) floats.
    """
    center_tile = mercantile.tile(lon, lat, zoom)
    # tile x,y of the lat/lon
    target_tile = mercantile.tile(lon, lat, zoom)
    # tile offsets relative to center (center is center_tile)
    dx = target_tile.x - center_tile.x
    dy = target_tile.y - center_tile.y
    # pixel position: center tile center + dx*tile_size + intra-tile pixel offset
    # compute intra-tile pixel offsets using mercantile.xy?
    # mercantile.ul and tile_to_bbox give latlon -> bbox; compute relative fraction.
    tb = mercantile.bounds(target_tile.x, target_tile.y, zoom)
    # bbox: west, south, east, north (lonMin, latMin, lonMax, latMax)
    lon_min, lat_min, lon_max, lat_max = tb.west, tb.south, tb.east, tb.north
    # fraction inside tile
    fx = (lon - lon_min) / (lon_max - lon_min) if (lon_max - lon_min) != 0 else 0.5
    fy = (lat_max - lat) / (lat_max - lat_min) if (lat_max - lat_min) != 0 else 0.5  # y flipped
    # pixel position in stitched image
    center_offset_x = (radius) * tile_size + tile_size * 0.5
    center_offset_y = (radius) * tile_size + tile_size * 0.5
    px = center_offset_x + dx * tile_size + fx * tile_size
    py = center_offset_y + dy * tile_size + fy * tile_size
    return px, py

# Minimal Imagen class (only methods used by imagenRunner)
class Imagen:
    def __init__(self, provider: str = "esri", cache_dir: str = "./tile_cache", timeout: int = 15, concurrency: int = 12, user_agent: str = "ImagenFast/1.0"):
        if provider.lower() == "osm":
            raise ValueError("OSM cannot be used programmatically; choose esri/google/bing/gibs.")
        self.provider = provider.lower()
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.concurrency = concurrency
        self.user_agent = user_agent
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ESRI tile URL
    def _esri_url(self, x, y, z):
        return f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    def downTile(self, x, y, z, key=None):
        url = self._esri_url(x, y, z) if self.provider == "esri" else None
        if url is None:
            raise RuntimeError("Provider not supported in this method.")
        r = requests.get(url, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Tile download failed ({r.status_code}): {url}")
        return Image.open(BytesIO(r.content)).convert("RGB")

    def getStitchedTiles(self, lat, lon, zoom, radius=1, key=None):
        center = mercantile.tile(lon, lat, zoom)
        tiles = []
        for dy in range(-radius, radius + 1):
            row = []
            for dx in range(-radius, radius + 1):
                tx = center.x + dx
                ty = center.y + dy
                row.append(self.downTile(tx, ty, zoom, key))
            tiles.append(row)
        row_imgs = [self._hstack(row) for row in tiles]
        final = self._vstack(row_imgs)
        return final

    # H/V stackers
    def _hstack(self, images):
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)
        out = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            out.paste(img, (x_offset, 0))
            x_offset += img.width
        return out

    def _vstack(self, images):
        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights)
        out = Image.new("RGB", (max_width, total_height))
        y_offset = 0
        for img in images:
            out.paste(img, (0, y_offset))
            y_offset += img.height
        return out
