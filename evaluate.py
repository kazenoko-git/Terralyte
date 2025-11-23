# evaluate.py
"""
Evaluate the saved multi-label model on a labeled JSON dataset.
Outputs per-label precision/recall/f1, and micro/macro f1 scores.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, hamming_loss

from inference import MODEL_PATH, TRAINING_JSON_PATH, Model

def evaluate(datasetPath: Path = TRAINING_JSON_PATH, modelPath: Path = MODEL_PATH):
    if not modelPath.exists():
        raise FileNotFoundError(f"Model file not found: {modelPath}")
    if not datasetPath.exists():
        raise FileNotFoundError(f"Dataset not found: {datasetPath}")

    m = Model(modelPath=modelPath, trainingJsonPath=datasetPath)
    if not m.isTrained:
        raise RuntimeError("Loaded model is not trained.")

    with open(datasetPath, 'r') as f:
        data = json.load(f)
    # prepare X and Y using the internal utilities
    X_list = []
    Y_list = []
    for item in data:
        X_list.append(m._makeFeatureVector(item))
        Y_list.append([n for n in item.get('qc_notes', []) if n in m.noteVocab])

    X = np.vstack(X_list)
    Y_true = m.mlb.transform(Y_list)  # shape (N, L)
    Y_pred_probs = m.tinyModel.predictProba(X)
    Y_pred = (Y_pred_probs >= m.labelThresholds).astype(int)

    # per-label precision/recall/f1
    p, r, f, sup = precision_recall_fscore_support(Y_true, Y_pred, zero_division=0)
    microF1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    macroF1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
    hamLoss = hamming_loss(Y_true, Y_pred)

    print("=== Multi-label Evaluation ===")
    print(f"Samples: {len(Y_true)}")
    print(f"Micro F1: {microF1:.4f}")
    print(f"Macro F1: {macroF1:.4f}")
    print(f"Hamming Loss: {hamLoss:.4f}\n")

    print("{:40s} {:6s} {:6s} {:6s} {:6s}".format("Label", "Prec", "Rec", "F1", "Support"))
    for i, label in enumerate(m.noteVocab):
        print("{:40s} {:6.2f} {:6.2f} {:6.2f} {:6d}".format(label, p[i]*100, r[i]*100, f[i]*100, int(sup[i])))
    # optionally return metrics for programmatic use
    return {
        "micro_f1": microF1,
        "macro_f1": macroF1,
        "hamming_loss": hamLoss,
        "per_label": {"label": m.noteVocab, "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": sup.tolist()}
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="path to labeled JSON dataset", default=str(TRAINING_JSON_PATH))
    p.add_argument("--model", help="path to model pickle", default=str(MODEL_PATH))
    args = p.parse_args()
    evaluate(Path(args.data), Path(args.model))
