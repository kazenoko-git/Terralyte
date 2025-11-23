#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imagen import Imagen, crop_center, latlon_to_pixel_in_stitched
from PIL import Image
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("lat", type=float)
parser.add_argument("lon", type=float)
parser.add_argument("zoom", type=int)
parser.add_argument("radius", type=int)
parser.add_argument("provider", type=str)
parser.add_argument("--crop", action="store_true", help="crop to target lat/lon")
parser.add_argument("--crop-size", type=int, default=640, help="final crop size (default 640)")
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

img = Imagen(provider=provider).getStitchedTiles(lat, lon, zoom, radius)

if crop:
    # compute pixel location within stitched image
    px, py = latlon_to_pixel_in_stitched(lat, lon, zoom, radius, tile_size=256)
    left = int(px - crop_size/2)
    top = int(py - crop_size/2)
    right = left + crop_size
    bottom = top + crop_size
    # clamp to image
    w, h = img.size
    left = max(0, min(left, w - crop_size))
    top = max(0, min(top, h - crop_size))
    right = left + crop_size
    bottom = top + crop_size
    img = img.crop((left, top, right, bottom))
else:
    # default center crop for convenience
    img = crop_center(img, crop_width=crop_size, crop_height=crop_size)

if out is None:
    out = f"tile_{lat}_{lon}_{zoom}.png"

img.save(out)
print(out)
