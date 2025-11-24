# tile_fetch.py
from pathlib import Path
from PIL import Image
import sys, os

# ensure imagen.py is importable
sys.path.append(str(Path(__file__).parent))

from imagen import Imagen, latlon_to_pixel_in_stitched, crop_center  # type: ignore

def get_stitched_tile(lat, lon, zoom, radius, provider, crop_size=640):
    img = Imagen(provider=provider).getStitchedTiles(lat, lon, zoom, radius)

    # always crop center (fallback)
    w, h = img.size
    if w < crop_size or h < crop_size:
        pad = Image.new("RGB", (max(w,crop_size), max(h,crop_size)), (0,0,0))
        pad.paste(img, ((pad.width - w)//2, (pad.height - h)//2))
        img = pad

    return crop_center(img, crop_size, crop_size)
