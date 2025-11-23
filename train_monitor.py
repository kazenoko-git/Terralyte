# train_monitor.py
"""
Training monitor & visualizer for Terralyte QC model.
Fixes:
- Normalizes human-readable AND camelCase labels
- Correct per-label evaluation
- Clean plotting
- Compatible with updated inference.py
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support

from inference import (
    Model,
    TRAINING_JSON_PATH,
    DISPLAY_LABELS,
    DISPLAY_TO_INTERNAL,
)


def normalize_labels(note_list, model: Model):
    """
    Converts human-readable OR camelCase labels into internal camelCase.
    Returns list of internal keys.
    """
    internal = []
    for n in note_list:
        # Already internal
        if n in model.noteVocab:
            internal.append(n)
            continue

        # Human-readable → internal
        if n in DISPLAY_TO_INTERNAL:
            internal.append(DISPLAY_TO_INTERNAL[n])
            continue

        # Case-insensitive fallback
        lowered = n.strip().lower()
        for internal_key, readable in DISPLAY_LABELS.items():
            if lowered == readable.lower():
                internal.append(internal_key)
                break

    return internal


def plot_label_frequency(training_list, model: Model, out_path: Path):
    """Plot distribution of training labels."""
    counts = {k: 0 for k in model.noteVocab}

    for item in training_list:
        notes = normalize_labels(item.get("qc_notes", []), model)
        for n in notes:
            if n in counts:
                counts[n] += 1

    labels_internal = list(model.noteVocab)
    labels_readable = [DISPLAY_LABELS[k] for k in labels_internal]
    freqs = [counts[k] for k in labels_internal]

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(labels_readable))
    plt.barh(y_pos, freqs)
    plt.yticks(y_pos, labels_readable)
    plt.xlabel("Training Frequency")
    plt.title("QC Label Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved label distribution to {out_path}")


def quick_evaluate(model: Model, training_list):
    """Evaluate model on training data (sanity check)."""
    X_list = []
    Y_true_list = []

    # Build true labels
    for item in training_list:
        X_list.append(model._makeFeatureVector(item))
        internal_notes = normalize_labels(item.get("qc_notes", []), model)
        Y_true_list.append(internal_notes)

    X = np.vstack(X_list)
    Y_true = model.mlb.fit_transform(Y_true_list)

    # Predictions
    Y_probs = model.tinyModel.predictProba(X)
    Y_pred = (Y_probs >= model.labelThresholds).astype(int)

    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    per_label = precision_recall_fscore_support(Y_true, Y_pred, zero_division=0)

    return {
        "micro": micro,
        "macro": macro,
        "precision": per_label[0],
        "recall": per_label[1],
        "f1": per_label[2],
        "support": per_label[3],
    }


def main():
    training_path = Path(TRAINING_JSON_PATH)

    if not training_path.exists():
        print("❌ Training JSON not found:", training_path)
        return

    # Load data
    with open(training_path, "r") as f:
        training_list = json.load(f)

    model = Model()

    # Plot distribution
    plot_label_frequency(training_list, model, training_path.parent / "label_distribution.png")

    # Train
    print("Training model (this may take a while)...")
    model.train(training_list, epochs=250, lr=5e-3, verbose=True)
    print("Training complete, saved model.")

    # Evaluate
    print("Running quick evaluation...")
    results = quick_evaluate(model, training_list)

    print(f"Micro F1: {results['micro']:.3f}")
    print(f"Macro F1: {results['macro']:.3f}")

    # Per label
    print()
    for i, internal_key in enumerate(model.noteVocab):
        readable = DISPLAY_LABELS[internal_key]
        print(
            f"{readable}: "
            f"F1={results['f1'][i]:.3f}, "
            f"P={results['precision'][i]:.3f}, "
            f"R={results['recall'][i]:.3f}, "
            f"support={results['support'][i]}"
        )


if __name__ == "__main__":
    main()
