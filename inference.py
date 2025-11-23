# inference.py
"""
Terralyte local multi-label QC notes model (uses YOLO image embeddings + metadata).
- Multilabel sigmoid MLP implemented in numpy (numpy + scikit-learn).
- Auto-train if qc_training_data.json is newer than saved model.
- Feature vector: [confidence, panel_count_norm, area_norm, capacity_norm, has_solar] + embedding (padded/truncated to EMB_DIM).
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# ---------------------------
# Config / Paths
# ---------------------------
PROJECT_ROOT = Path("/Users/ivansamuel/PycharmProjects/Terralyte")
TRAINING_JSON_PATH = PROJECT_ROOT / "qc_training_data.json"
MODEL_PATH = PROJECT_ROOT / "solarQcModel.pkl"
AUTO_TRAIN_ON_START = True

# Embedding config
EMB_DIM = 512  # default embedding vector length to pad/truncate to

# ---------------------------
# Human-readable labels
# ---------------------------
DISPLAY_LABELS = {
    "highConfidenceDetection": "High Confidence Detection",
    "moderateConfidenceDetection": "Moderate Confidence Detection",
    "lowConfidenceNeedsVerification": "Low Confidence — Needs Verification",
    "clearRoofView": "Clear Roof View",
    "obscuredOrUnclearView": "Obscured or Unclear View",
    "largeCommercialInstallation": "Large Commercial Installation",
    "residentialSystemDetected": "Residential System Detected",
    "distinctModuleGridVisible": "Distinct Module Grid Visible",
    "significantPanelAreaCoverage": "Significant Panel Area Coverage",
    "possibleFalsePositiveRoofLikePattern": "Possible False Positive — Roof-Like Pattern",
    "shadowsMayCauseMissedModules": "Shadows May Cause Missed Modules",
    "partialOcclusionDetected": "Partial Occlusion Detected",
    "lowResolutionImage": "Low Resolution Image",
    "glareOrSunReflection": "Glare or Sun Reflection",
    "tiltedPanelsDetected": "Tilted Panels Detected",
    "groundMountedArrayDetected": "Ground-Mounted Array Detected",
    "trackerSystemsDetected": "Tracker Systems Detected",
    "bifacialPanelsDetected": "Bifacial Panels Detected",
    "nearbyObjectsObstructing": "Nearby Objects Obstructing Panels",
    "possibleDebrisOrDamage": "Possible Debris or Damage",
    "multipleRoofSegments": "Multiple Roof Segments",
    "inconsistentModuleSpacing": "Inconsistent Module Spacing",
    "inverterOrEquipmentVisible": "Inverter or Equipment Visible",
    "recentInstallationLikely": "Recent Installation Likely",
    "historicalChangeDetected": "Historical Change Detected"
}
DISPLAY_TO_INTERNAL = {v: k for k, v in DISPLAY_LABELS.items()}

# ---------------------------
# Helpers
# ---------------------------
def safeGet(d: Dict, k: str, default=0.0):
    try:
        return float(d.get(k, default))
    except:
        return default

def pad_or_truncate_embedding(arr, dim=EMB_DIM):
    arr = np.asarray(arr, dtype=float) if arr is not None else np.zeros((0,), dtype=float)
    if arr.size >= dim:
        return arr[:dim]
    else:
        out = np.zeros(dim, dtype=float)
        out[:arr.size] = arr
        return out

# ---------------------------
# Tiny MLP
# ---------------------------
class TinyMultiLabelMLP:
    def __init__(self, inputDim, hidden=128, outDim=25, seed=42):
        self.rng = np.random.RandomState(seed)
        self.inputDim = int(inputDim)
        self.hidden = int(hidden)
        self.outDim = int(outDim)
        # Xavier-ish init scaled appropriately
        self.W1 = self.rng.normal(0, np.sqrt(2.0 / (self.inputDim + self.hidden)), (self.hidden, self.inputDim))
        self.b1 = np.zeros(self.hidden)
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / (self.hidden + self.outDim)), (self.outDim, self.hidden))
        self.b2 = np.zeros(self.outDim)

    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x):
        h = self.relu(x @ self.W1.T + self.b1)
        logits = h @ self.W2.T + self.b2
        probs = self.sigmoid(logits)
        return probs, logits, h

    def predictProba(self, x):
        x = np.atleast_2d(x)
        return self.forward(x)[0]

    def predict(self, x, thresh=0.5):
        return (self.predictProba(np.atleast_2d(x)) >= thresh).astype(int)

    def fit(self, X, Y, lr=5e-3, epochs=300, batch=32, verbose=False):
        N = X.shape[0]
        for epoch in range(epochs):
            idx = self.rng.permutation(N)
            epoch_loss = 0.0
            for start in range(0, N, batch):
                b = idx[start:start+batch]
                xb = X[b]
                yb = Y[b]
                probs, logits, h = self.forward(xb)
                eps = 1e-9
                loss = -(yb * np.log(probs + eps) + (1 - yb) * np.log(1 - probs + eps)).mean()
                epoch_loss += loss * len(b)

                dlog = (probs - yb) / len(b)
                dW2 = dlog.T @ h
                db2 = dlog.sum(axis=0)
                dh = dlog @ self.W2
                dh[h <= 0] = 0
                dW1 = dh.T @ xb
                db1 = dh.sum(axis=0)

                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"[MLP] epoch {epoch+1}/{epochs} avgLoss={(epoch_loss / N):.6f}")

# ---------------------------
# Model
# ---------------------------
class Model:
    def __init__(self, modelPath=MODEL_PATH, trainingJsonPath=TRAINING_JSON_PATH, emb_dim=EMB_DIM):
        self.modelPath = Path(modelPath)
        self.trainingJsonPath = Path(trainingJsonPath)
        self.noteVocab = self._makeVocab()
        self.mlb = MultiLabelBinarizer(classes=self.noteVocab)
        self.tinyModel: Optional[TinyMultiLabelMLP] = None
        self.isTrained = False
        self.labelThresholds = np.ones(len(self.noteVocab)) * 0.5
        self.emb_dim = int(emb_dim)

        if self.modelPath.exists():
            try:
                self.loadModel()
            except Exception as e:
                print("[Model] load failure:", e)
                self.isTrained = False

        if AUTO_TRAIN_ON_START:
            try:
                self.autoTrainIfNeeded()
            except Exception as e:
                print("[Model] auto-train failed:", e)

    def _makeVocab(self):
        return list(DISPLAY_LABELS.keys())

    # ---------------------------
    # Feature vector includes embedding (robust)
    # ---------------------------
    def _makeFeatureVector(self, d: Dict):
        confidence = safeGet(d, "confidence")
        panels = safeGet(d, "panel_count") / 100.0
        area = safeGet(d, "area_sqm") / 200.0
        capacity = safeGet(d, "capacity_kw") / 50.0
        has_solar = 1.0 if d.get("has_solar", False) else 0.0

        meta = np.array([confidence, panels, area, capacity, has_solar], dtype=float)

        emb_raw = d.get("embedding", [])
        # flatten nested
        if isinstance(emb_raw, (list, tuple)) and emb_raw and isinstance(emb_raw[0], (list, tuple, np.ndarray)):
            emb_raw = np.asarray(emb_raw).flatten().tolist()
        emb = pad_or_truncate_embedding(emb_raw, dim=self.emb_dim)
        if np.linalg.norm(emb) > 0:
            emb = emb / (np.linalg.norm(emb) + 1e-9)
        return np.concatenate([meta, emb]).astype(float)

    def _alignFeatureDimToModel(self, fv: np.ndarray) -> np.ndarray:
        if self.tinyModel is None:
            return fv
        expected = int(self.tinyModel.inputDim)
        current = int(fv.size)
        if current == expected:
            return fv
        if current < expected:
            pad = np.zeros(expected - current, dtype=float)
            fv2 = np.concatenate([fv, pad])
            print(f"[Model] WARN: input dim mismatch (got {current}, expected {expected}) — padding with {pad.size} zeros.")
            return fv2
        else:
            meta = fv[:5]
            emb = fv[5:]
            emb_trunc = emb[: max(0, expected - 5)]
            fv2 = np.concatenate([meta, emb_trunc])
            print(f"[Model] WARN: input dim mismatch (got {current}, expected {expected}) — truncating embedding to fit.")
            return fv2

    # ---------------------------
    # Rule-based fallback
    # ---------------------------
    def _ruleBasedNotes(self, data: Dict) -> List[str]:
        conf = safeGet(data, "confidence")
        count = safeGet(data, "panel_count")
        area = safeGet(data, "area_sqm")
        notes = []

        if conf >= 0.85:
            notes.append("highConfidenceDetection")
        elif conf >= 0.65:
            notes.append("moderateConfidenceDetection")
        else:
            notes.append("lowConfidenceNeedsVerification")

        if conf >= 0.75:
            notes.append("clearRoofView")
        else:
            notes.append("obscuredOrUnclearView")

        if count >= 20:
            notes.append("largeCommercialInstallation")
        elif count >= 8:
            notes.append("residentialSystemDetected")

        if count >= 4 and conf >= 0.6:
            notes.append("distinctModuleGridVisible")

        if area >= 100:
            notes.append("significantPanelAreaCoverage")

        return notes[:8]

    # ---------------------------
    # Public API
    # ---------------------------
    def generateQcNotes(self, detectionData: Dict, threshold: float = 0.5) -> List[str]:
        fv = self._makeFeatureVector(detectionData)
        if self.tinyModel is not None:
            fv = self._alignFeatureDimToModel(fv)
        if not self.isTrained or self.tinyModel is None:
            return [DISPLAY_LABELS[k] for k in self._ruleBasedNotes(detectionData)]

        try:
            probs = self.tinyModel.predictProba(fv.reshape(1, -1))[0]
        except Exception as e:
            print("[Model] prediction failed — falling back to rule-based notes. Error:", e)
            internal = self._ruleBasedNotes(detectionData)
            return [DISPLAY_LABELS[k] for k in internal]

        chosen = [self.noteVocab[i] for i, p in enumerate(probs) if p >= self.labelThresholds[i]]
        if len(chosen) == 0:
            topIdx = np.argsort(-probs)[:2]
            chosen = [self.noteVocab[i] for i in topIdx]
        return [DISPLAY_LABELS[k] for k in chosen]

    # ---------------------------
    # Training helpers
    # ---------------------------
    def _normalizeNote(self, n: str):
        if n in self.noteVocab:
            return n
        if n in DISPLAY_TO_INTERNAL:
            return DISPLAY_TO_INTERNAL[n]
        for key, readable in DISPLAY_LABELS.items():
            if n.lower() == readable.lower():
                return key
        return None

    def _prepareTrainingData(self, trainingList: List[Dict]):
        X_list = []
        Y_list = []
        for item in trainingList:
            fv = self._makeFeatureVector(item)
            X_list.append(fv)
            notes = item.get("qc_notes", [])
            norm = [self._normalizeNote(x) for x in notes]
            Y_list.append([n for n in norm if n in self.noteVocab])
        X = np.vstack(X_list).astype(np.float32)
        Ybin = self.mlb.fit_transform(Y_list)
        return X, Ybin

    def train(self, trainingList: List[Dict], epochs: int = 300, lr: float = 5e-3, verbose: bool = False):
        if len(trainingList) < 4:
            raise ValueError("Need at least 4 training samples")
        X, Y = self._prepareTrainingData(trainingList)
        inDim = X.shape[1]
        outDim = Y.shape[1]
        self.tinyModel = TinyMultiLabelMLP(inputDim=inDim, hidden=128, outDim=outDim)
        self.tinyModel.fit(X, Y, lr=lr, epochs=epochs, batch=16, verbose=verbose)
        labelPosRates = Y.mean(axis=0)
        self.labelThresholds = np.clip(0.5 * (1.0 + labelPosRates), 0.2, 0.7)
        self.isTrained = True
        self.saveModel()

    # ---------------------------
    # Persistence
    # ---------------------------
    def saveModel(self):
        payload = {
            'isTrained': self.isTrained,
            'noteVocab': self.noteVocab,
            'labelThresholds': self.labelThresholds,
            'emb_dim': self.emb_dim,
            'mlp': {
                'W1': self.tinyModel.W1 if self.tinyModel is not None else None,
                'b1': self.tinyModel.b1 if self.tinyModel is not None else None,
                'W2': self.tinyModel.W2 if self.tinyModel is not None else None,
                'b2': self.tinyModel.b2 if self.tinyModel is not None else None,
                'inputDim': self.tinyModel.inputDim if self.tinyModel is not None else None,
                'hidden': self.tinyModel.hidden if self.tinyModel is not None else None,
                'outDim': self.tinyModel.outDim if self.tinyModel is not None else None,
            }
        }
        with open(self.modelPath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[Model] saved to {self.modelPath}")

    def loadModel(self):
        with open(self.modelPath, 'rb') as f:
            payload = pickle.load(f)
        self.isTrained = payload.get('isTrained', False)
        self.noteVocab = payload.get('noteVocab', self._makeVocab())
        self.mlb = MultiLabelBinarizer(classes=self.noteVocab)
        self.labelThresholds = np.array(payload.get('labelThresholds', [0.5]*len(self.noteVocab)))
        self.emb_dim = int(payload.get('emb_dim', self.emb_dim))
        mlp = payload.get('mlp', None)
        if mlp and mlp.get('W1') is not None:
            mlpObj = TinyMultiLabelMLP(inputDim=mlp['inputDim'], hidden=mlp['hidden'], outDim=mlp['outDim'])
            mlpObj.W1 = mlp['W1']
            mlpObj.b1 = mlp['b1']
            mlpObj.W2 = mlp['W2']
            mlpObj.b2 = mlp['b2']
            self.tinyModel = mlpObj
            self.isTrained = True
        else:
            self.tinyModel = None
            self.isTrained = False
        print(f"[Model] loaded {self.modelPath}; trained={self.isTrained}")

    # ---------------------------
    # Auto-train when training file is newer than model
    # ---------------------------
    def autoTrainIfNeeded(self):
        if not self.trainingJsonPath.exists():
            return False
        trainingTs = self.trainingJsonPath.stat().st_mtime
        if self.modelPath.exists():
            modelTs = self.modelPath.stat().st_mtime
        else:
            modelTs = 0
        if trainingTs > modelTs:
            print("[Model] training JSON is newer than saved model. Starting auto-train...")
            with open(self.trainingJsonPath, 'r') as f:
                trainingList = json.load(f)
            self.train(trainingList, epochs=250, lr=5e-3, verbose=True)
            return True
        return False

# ---------------------------
# Module-level convenience
# ---------------------------
_globalModel = Model()

def generateQcNotes(detectionData: Dict) -> List[str]:
    return _globalModel.generateQcNotes(detectionData)

def trainModelFromFile(jsonPath: str = str(TRAINING_JSON_PATH)):
    with open(jsonPath, 'r') as f:
        data = json.load(f)
    _globalModel.train(data)
