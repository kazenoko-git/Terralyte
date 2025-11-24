#!/usr/bin/env python3
"""
run_model.py — orchestrator for Terralyte backend

CLI contract preserved for Tauri:
    python run_model.py <imageInput> <weights> [--lat <lat> --lon <lon> --zoom <z> --capture-date <date>]

Inputs:
 - imageInput: path to .b64, path to image, or raw base64 string
 - weights: path to verifier2.pt (we recommend Terralyte/models/verifier2.pt)
"""

import sys
import json
import time
import traceback
from pathlib import Path

# local modules
from preprocess import decode_input
from yolo_engine import load_yolo_model, infer_with_embedding
from postprocess import aggregate_predictions
from overlay import render_overlay
from qc_model import Model as QCModel, generateQcNotes
from utils import nowStr, save_json_atomic

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_DIR = PROJECT_ROOT / "detections"
AUDIT_DIR = PROJECT_ROOT / "audit_overlays"
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# defaults
DEFAULT_IMG_SIZE = 1024
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_ZOOM = 18
AVG_PANEL_AREA_SQM = 1.7

def processImage(imageInputArg, weightsPath,
                 lat=None, lon=None, zoom=None, imgsz=DEFAULT_IMG_SIZE, conf=DEFAULT_CONF,
                 capture_date: str = None):
    try:
        resolved_path, pil_img, meta_sidecar = decode_input(imageInputArg, PROJECT_ROOT)
    except Exception as e:
        raise RuntimeError(f"Failed to load image input: {e}")

    # merge sidecar
    if meta_sidecar:
        try:
            p = Path(meta_sidecar)
            if p.exists():
                meta = json.load(open(p, "r"))
                lat = lat if lat is not None else meta.get("lat", lat)
                lon = lon if lon is not None else meta.get("lon", lon)
                zoom = zoom if zoom is not None else meta.get("zoom", zoom)
                capture_date = capture_date if capture_date is not None else meta.get("capture_date", capture_date)
        except Exception:
            pass

    if lat is None: lat = 0.0
    if lon is None: lon = 0.0
    if zoom is None: zoom = DEFAULT_ZOOM
    if capture_date is None:
        capture_date = nowStr()

    # load model (cached)
    model, device = load_yolo_model(weightsPath)

    # inference
    dets, embedding = infer_with_embedding(model, device, resolved_path, imgsz=imgsz, conf=conf)

    # image dims
    try:
        w, h = pil_img.size
    except Exception:
        w, h = imgsz, imgsz

    # aggregate
    agg = aggregate_predictions(dets, w, h, lat, zoom)

    # QC notes
    try:
        qc_model = QCModel()
        qc_notes = qc_model.generateQcNotes(agg)
    except Exception:
        # fallback to module-level
        qc_notes = generateQcNotes(agg)

    qc_status = "verifiable" if agg["has_solar"] else "no_solar_detected"

    # overlay path
    overlay_path = AUDIT_DIR / f"audit_overlay_{int(time.time()*1000)}.png"
    try:
        confidences = [float(d.get("confidence", 0.0)) for d in dets] if dets else []
        render_overlay(pil_img, agg["boxes"], agg["masks"], confidences, str(overlay_path))
    except Exception:
        overlay_path = ""

    # append training — safe dedupe (we will do a gentle append)
    # Keep training append simple: store relevant summary
    try:
        training_file = PROJECT_ROOT / "qc_training_data.json"
        tf_exists = training_file.exists()
        if tf_exists:
            data = json.load(open(training_file, "r"))
            if not isinstance(data, list):
                data = []
        else:
            data = []
        # create summary
        summary = {
            "timestamp": nowStr(),
            "confidence": agg["confidence"],
            "panel_count": agg["panel_count"],
            "area_sqm": agg["area_sqm"],
            "capacity_kw": agg["capacity_kw"],
            "has_solar": agg["has_solar"],
            "qc_notes": qc_notes,
            "embedding": embedding
        }
        # de-dupe: compare lightweight fingerprint
        def fp(s):
            return f"{s.get('confidence',0)}|{s.get('panel_count',0)}|{s.get('area_sqm',0)}"
        if all(fp(entry) != fp(summary) for entry in data):
            data.append(summary)
            with open(training_file, "w") as f:
                json.dump(data, f, indent=2)
            appended = True
        else:
            appended = False
    except Exception:
        appended = False

    # pick bbox_or_mask for JSON
    if agg["masks"]:
        bbox_or_mask = {"masks_present": True, "num_masks": len(agg["masks"])}
    else:
        bbox_or_mask = agg["boxes"]

    sample_id = int(time.time()*1000) % (10**12)
    out = {
        "sample_id": sample_id,
        "lat": float(lat),
        "lon": float(lon),
        "has_solar": agg["has_solar"],
        "confidence": agg["confidence"],
        "panel_count_Est": agg["panel_count"],
        "pv_area_sqm_est": agg["area_sqm"],
        "capacity_kw_est": agg["capacity_kw"],
        "qc_status": qc_status,
        "qc_notes": qc_notes,
        "bbox_or_mask": bbox_or_mask,
        "clusters": agg.get("clusters", []),
        "mpp": agg.get("mpp", None),
        "audit_overlay_path": str(overlay_path) if overlay_path else "",
        "image_metadata": {
            "source": Path(resolved_path).name if resolved_path else None,
            "capture_date": capture_date,
            "audit_overlay_path": str(overlay_path) if overlay_path else ""
        },
        "appended_to_training": appended,
        "timestamp": nowStr()
    }

    # save detection JSON
    try:
        out_path = DETECTIONS_DIR / f"detection_{sample_id}.json"
        save_json_atomic(out_path, out)
    except Exception:
        pass

    # print single JSON object (frontend expects this)
    print(json.dumps(out))
    return out


# CLI ENTRY
def main():
    if len(sys.argv) < 3:
        print("Usage: python run_model.py <imageInput> <weights> [--lat <lat> --lon <lon> --zoom <z> --capture-date YYYY-MM-DD]", file=sys.stderr)
        sys.exit(1)

    raw_input = sys.argv[1]
    weightsPath = sys.argv[2]

    lat = None
    lon = None
    zoom = None
    capture_date = None

    i = 3
    while i < len(sys.argv):
        a = sys.argv[i]
        if a in ("--lat", "-lat"):
            lat = float(sys.argv[i+1]); i += 2; continue
        if a in ("--lon", "-lon"):
            lon = float(sys.argv[i+1]); i += 2; continue
        if a in ("--zoom", "-z"):
            zoom = int(sys.argv[i+1]); i += 2; continue
        if a in ("--capture-date",):
            capture_date = sys.argv[i+1]; i += 2; continue
        i += 1

    try:
        processImage(raw_input, weightsPath, lat=lat, lon=lon, zoom=zoom, imgsz=DEFAULT_IMG_SIZE, conf=DEFAULT_CONF, capture_date=capture_date)
    except Exception as e:
        print("[run_model] ERROR during processing:", e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
