# overlay.py
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def render_overlay(pil_img, boxes, masks, confidences, out_path):
    img = pil_img.convert("RGBA")
    ovl = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(ovl, "RGBA")

    # masks
    for m in masks:
        mk = Image.fromarray((m*255).astype("uint8"),"L").resize(img.size)
        green = Image.new("RGBA", img.size, (0,200,100,120))
        ovl.paste(green, (0,0), mk)

    # boxes
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        draw.rectangle([x1,y1,x2,y2], outline=(255,255,0,255), width=2)
        label = f"{confidences[i]*100:.0f}%"
        draw.rectangle([x1,y1-16,x1+60,y1], fill=(0,0,0,200))
        draw.text((x1+3,y1-15), label, fill=(255,255,255,255))

    out = Image.alpha_composite(img, ovl)
    out = out.filter(ImageFilter.SHARPEN)
    out.save(out_path)
