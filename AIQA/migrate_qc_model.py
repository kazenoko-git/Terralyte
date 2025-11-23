# migrate_qc_model.py
"""
Migrates an existing solarQcModel.pkl to the new human-readable QC label system.
Keeps:
- learned weights
- thresholds
- classifier structure

Updates:
- noteVocab (ensures it matches new internal keys)
- adds DISPLAY_LABELS mapping into the pickle
- ensures labelThresholds has correct length
"""

import pickle
from pathlib import Path

MODEL_PATH = Path("/Users/ivansamuel/PycharmProjects/Terralyte/solarQcModel.pkl")

# New target internal vocab (MUST match inference.py order exactly)
NEW_VOCAB = [
    "highConfidenceDetection",
    "moderateConfidenceDetection",
    "lowConfidenceNeedsVerification",
    "clearRoofView",
    "obscuredOrUnclearView",
    "largeCommercialInstallation",
    "residentialSystemDetected",
    "distinctModuleGridVisible",
    "significantPanelAreaCoverage",
    "possibleFalsePositiveRoofLikePattern",
    "shadowsMayCauseMissedModules",
    "partialOcclusionDetected",
    "lowResolutionImage",
    "glareOrSunReflection",
    "tiltedPanelsDetected",
    "groundMountedArrayDetected",
    "trackerSystemsDetected",
    "bifacialPanelsDetected",
    "nearbyObjectsObstructing",
    "possibleDebrisOrDamage",
    "multipleRoofSegments",
    "inconsistentModuleSpacing",
    "inverterOrEquipmentVisible",
    "recentInstallationLikely",
    "historicalChangeDetected"
]

# Human-readable mapping
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

def migrate():
    if not MODEL_PATH.exists():
        print("❌ Model file not found:", MODEL_PATH)
        return

    print("🔧 Loading existing model...")
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    # Update vocabulary
    old_vocab = payload.get("noteVocab", None)
    print("Old vocab length:", len(old_vocab) if old_vocab else "None")

    payload["noteVocab"] = NEW_VOCAB

    # Resize labelThresholds if needed
    old_thresh = payload.get("labelThresholds", [])
    if len(old_thresh) != len(NEW_VOCAB):
        print("⚠ Resizing labelThresholds...")
        new_thresh = [0.5] * len(NEW_VOCAB)
        for i in range(min(len(old_thresh), len(new_thresh))):
            new_thresh[i] = old_thresh[i]
        payload["labelThresholds"] = new_thresh

    # Attach human-readable mapping for debugging
    payload["displayLabels"] = DISPLAY_LABELS

    # Save migrated model
    print("💾 Saving migrated model...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print("✅ Migration complete!")

if __name__ == "__main__":
    migrate()
