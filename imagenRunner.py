#!/usr/bin/env python3
import sys, os
from pathlib import Path
from PIL import Image

# Import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imagen import Imagen, latlon_to_pixel_in_stitched, crop_center

import argparse

# ------------------------------------------------------------
# CLI PARSER
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("lat", type=float)
parser.add_argument("lon", type=float)
parser.add_argument("zoom", type=int)
parser.add_argument("radius", type=int)
parser.add_argument("provider", type=str)

parser.add_argument("--crop", action="store_true")
parser.add_argument("--crop-size", type=int, default=640)
parser.add_argument("--out", type=str, default=None)

args = parser.parse_args()

lat = args.lat
lon = args.lon
zoom = args.zoom
radius = args.radius
provider = args.provider.lower()
crop = args.crop
crop_size = args.crop_size
out_name = args.out

# ------------------------------------------------------------
# STITCHED TILE FETCH
# ------------------------------------------------------------
try:
    img = Imagen(provider=provider).getStitchedTiles(lat, lon, zoom, radius)
except Exception as e:
    print(f"ERROR: Failed to fetch stitched tiles: {e}")
    sys.exit(1)

# ------------------------------------------------------------
# CROP LOGIC (USING FIXED, CORRECT WEBMERCATOR MATH)
# ------------------------------------------------------------
if crop:
    try:
        # Compute correct global pixel → stitched pixel mapping
        px, py = latlon_to_pixel_in_stitched(lat, lon, zoom, radius, tile_size=256)

        left   = int(px - crop_size/2)
        top    = int(py - crop_size/2)
        right  = left + crop_size
        bottom = top + crop_size

        # Clamp crop bounds to valid image region
        w, h = img.size
        if left < 0:             left = 0; right = crop_size
        if top < 0:              top = 0; bottom = crop_size
        if right > w:            right = w; left = w - crop_size
        if bottom > h:           bottom = h; top = h - crop_size

        # Final crop
        img = img.crop((left, top, right, bottom))

    except Exception as e:
        print(f"ERROR: crop failed: {e}")
        sys.exit(1)

else:
    # Standard fallback center crop
    img = crop_center(img, crop_width=crop_size, crop_height=crop_size)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------
if out_name is None:
    out_name = f"tile_{lat}_{lon}_{zoom}.png"

try:
    img.save(out_name)
except Exception as e:
    print(f"ERROR: failed to save output image: {e}")
    sys.exit(1)

print(out_name)
