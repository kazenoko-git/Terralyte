# preprocess.py
import base64, io, os, tempfile
from pathlib import Path
from PIL import Image

def is_probable_b64(s: str):
    if len(s) < 200: return False
    valid = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
    return all(c in valid for c in s[:200])

def decode_input(arg: str, project_root: Path):
    p = Path(arg)

    # 1. If file.b64
    if p.suffix.lower() == ".b64" and p.exists():
        data = p.read_text().strip()
        if data.startswith("data:"): data = data.split(",",1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")
        fd, tmp = tempfile.mkstemp(suffix=".png", dir=str(project_root))
        os.close(fd)
        img.save(tmp)
        return tmp, img, str(p.with_suffix(".meta.json"))

    # 2. Raw base64 string
    if is_probable_b64(arg):
        s = arg
        if s.startswith("data:"): s = s.split(",",1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGBA")
        fd, tmp = tempfile.mkstemp(suffix=".png", dir=str(project_root))
        os.close(fd)
        img.save(tmp)
        return tmp, img, None

    # 3. Normal file path
    if p.exists():
        img = Image.open(p).convert("RGBA")
        return str(p), img, str(p.with_suffix(".meta.json"))

    raise RuntimeError(f"Unable to decode input: {arg}")
