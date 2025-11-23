# debug_inference.py
import json
import numpy as np
from pathlib import Path
from inference import Model, TRAINING_JSON_PATH

p = Path(TRAINING_JSON_PATH)
if not p.exists():
    print("No training JSON found at", p)
    raise SystemExit(1)

with open(p, "r") as f:
    data = json.load(f)

m = Model()  # will load model pickle
print("Loaded model trained:", m.isTrained)
print("Vocab length:", len(m.noteVocab))
print("First 5 vocab (internal -> human):")
for k in m.noteVocab[:5]:
    from inference import DISPLAY_LABELS
    print(" ", k, "->", DISPLAY_LABELS.get(k, k))

# show 5 sample feature vectors and labels
for i, item in enumerate(data[:5]):
    fv = m._makeFeatureVector(item)
    y_true = [m._normalizeNote(n) for n in item.get("qc_notes", [])]
    y_true = [x for x in y_true if x]
    print(f"\nSample {i}:")
    print(" image:", item.get("image_path"))
    print(" features:", fv.tolist())
    print(" true internal labels:", y_true)
    # prediction
    if m.tinyModel is not None:
        probs = m.tinyModel.predictProba(fv.reshape(1, -1))[0]
        print(" probs (first 8):", probs[:8].tolist())
        pred = (probs >= m.labelThresholds).astype(int)
        print(" pred mask (first 8):", pred[:8].tolist())
    else:
        print(" model not trained yet")
