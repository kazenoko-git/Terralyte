#!/usr/bin/env python3
import sys, os
from pathlib import Path
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imagen import Imagen, latlon_to_pixel_in_stitched, crop_center

import argparse

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
radius = max(0, args.radius)   # sanitize
provider = args.provider.lower()
crop = args.crop
crop_size = args.crop_size
out_name = args.out

try:
    img = Imagen(provider=provider).getStitchedTiles(lat, lon, zoom, radius)
except Exception as e:
    print(f"ERROR: Failed to fetch stitched tiles: {e}")
    sys.exit(1)

if crop:
    try:
        px, py = latlon_to_pixel_in_stitched(lat, lon, zoom, radius, tile_size=256)

        # compute crop box in stitched-image coords
        left = int(round(px - crop_size / 2.0))
        top = int(round(py - crop_size / 2.0))
        right = left + crop_size
        bottom = top + crop_size

        w, h = img.size

        # If crop box extends outside stitched image, we create a padded canvas
        if left < 0 or top < 0 or right > w or bottom > h:
            canvas = Image.new("RGB", (max(w, crop_size), max(h, crop_size)), (0, 0, 0))
            # paste stitched image centered so that crop coordinates align
            paste_x = max(0, (canvas.width - w) // 2)
            paste_y = max(0, (canvas.height - h) // 2)
            canvas.paste(img, (paste_x, paste_y))
            # adjust crop coords because of paste offset
            left += paste_x
            right += paste_x
            top += paste_y
            bottom += paste_y
            img = canvas
            w, h = img.size

        # clamp to boundaries
        left = max(0, min(left, w - crop_size))
        top = max(0, min(top, h - crop_size))
        right = left + crop_size
        bottom = top + crop_size

        img = img.crop((left, top, right, bottom))

    except Exception as e:
        print(f"ERROR: crop failed: {e}")
        sys.exit(1)
else:
    # fallback center crop, safe if stitched smaller than target (will pad)
    try:
        w, h = img.size
        if w < crop_size or h < crop_size:
            canvas = Image.new("RGB", (max(w, crop_size), max(h, crop_size)), (0, 0, 0))
            canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
            img = canvas
        img = crop_center(img, crop_width=crop_size, crop_height=crop_size)
    except Exception:
        img = crop_center(img, crop_width=crop_size, crop_height=crop_size)

if out_name is None:
    out_name = f"tile_{lat}_{lon}_{zoom}.png"

try:
    img.save(out_name)
except Exception as e:
    print(f"ERROR: failed to save output image: {e}")
    sys.exit(1)

print(out_name)
