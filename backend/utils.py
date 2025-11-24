# utils.py
from pathlib import Path
import time
import hashlib
import json

def nowStr():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def shaDetection(path: str, summary: dict):
    s = f"{path}|{summary.get('panel_count')}|{summary.get('area_sqm')}|{summary.get('capacity_kw')}"
    return hashlib.sha256(s.encode()).hexdigest()

def save_json_atomic(path: Path, data):
    temp = path.with_suffix(".tmp")
    with open(temp, "w") as f:
        json.dump(data, f, indent=2)
    temp.replace(path)
