#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagen import Imagen
from pathlib import Path
from PIL import Image
import argparse
import math

# ==============================
# Web Mercator helpers
# ==============================
TILE_SIZE = 256

def latlon_to_global_pixels(lat, lon, zoom):
    """Returns global pixel coordinates in Web Mercator at given zoom."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    y = (
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
        * TILE_SIZE
    )
    return x, y

def stitched_offset_pixels(lat, lon, zoom, radius):
    """
    RETURNS:
      (px, py) coordinates of the lat/lon inside the stitched image.
    """
    gx, gy = latlon_to_global_pixels(lat, lon, zoom)

    tile_x = gx / TILE_SIZE
    tile_y = gy / TILE_SIZE

    tile_x_int = math.floor(tile_x)
    tile_y_int = math.floor(tile_y)

    # fractional offset within the tile
    dx = tile_x - tile_x_int
    dy = tile_y - tile_y_int

    # stitched is (2*radius+1) tiles each direction
    px = (radius + dx) * TILE_SIZE
    py = (radius + dy) * TILE_SIZE
    return px, py


# ==============================
# CLI
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("lat", type=float)
parser.add_argument("lon", type=float)
parser.add_argument("zoom", type=int)
parser.add_argument("radius", type=int)
parser.add_argument("provider", type=str)
parser.add_argument("--crop", action="store_true", help="crop to target lat/lon")
parser.add_argument("--crop-size", type=int, default=640, help="size of final crop")
parser.add_argument("--out", type=str, default=None, help="output filename")
args = parser.parse_args()

lat = args.lat
lon = args.lon
zoom = args.zoom
radius = args.radius
provider = args.provider

crop = args.crop
crop_size = args.crop_size
out = args.out

# Fetch stitched tiles using your Imagen class
img = Imagen(provider=provider).getStitchedTiles(lat, lon, zoom, radius)
w, h = img.size

if crop:
    # Compute accurate pixel position of lat/lon within stitched image
    px, py = stitched_offset_pixels(lat, lon, zoom, radius)

    # Define crop bounding box
    left = int(px - crop_size / 2)
    top = int(py - crop_size / 2)

    # Clamp
    left = max(0, min(left, w - crop_size))
    top = max(0, min(top, h - crop_size))

    right = left + crop_size
    bottom = top + crop_size

    img = img.crop((left, top, right, bottom))
else:
    # default perfect center crop
    cx = w // 2
    cy = h // 2
    left = cx - crop_size // 2
    top = cy - crop_size // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))

# output
if out is None:
    out = f"tile_{lat}_{lon}_{zoom}.png"

img.save(out)
print(out)
