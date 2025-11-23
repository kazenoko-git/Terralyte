#!/usr/bin/env python3
# generate_balanced_multilabel_qc.py
"""
Balanced multilabel QC dataset generator (Ultimate 2.0)
- Produces N_per_label samples where each label is the primary target.
- Each sample contains 3-5 labels (primary + 2-4 compatible co-labels).
- Writes qc_training_data.json into your Terralyte project.
"""

import json
import random
import uuid
from pathlib import Path
from math import sqrt

OUT = Path("/Users/ivansamuel/PycharmProjects/Terralyte/qc_training_data.json")

# Human-readable labels (must match inference.py)
DISPLAY = {
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

LABEL_KEYS = list(DISPLAY.keys())

random.seed(42)

# Compatibility map: for each label, list of labels that make sense as co-labels.
# Keep groups tight to avoid contradictions.
COMPAT = {
    "highConfidenceDetection": ["clearRoofView", "distinctModuleGridVisible", "recentInstallationLikely"],
    "moderateConfidenceDetection": ["clearRoofView", "distinctModuleGridVisible", "partialOcclusionDetected"],
    "lowConfidenceNeedsVerification": ["obscuredOrUnclearView", "lowResolutionImage", "possibleFalsePositiveRoofLikePattern"],
    "clearRoofView": ["highConfidenceDetection", "distinctModuleGridVisible", "recentInstallationLikely"],
    "obscuredOrUnclearView": ["lowConfidenceNeedsVerification", "partialOcclusionDetected", "nearbyObjectsObstructing"],
    "largeCommercialInstallation": ["significantPanelAreaCoverage", "groundMountedArrayDetected", "trackerSystemsDetected"],
    "residentialSystemDetected": ["distinctModuleGridVisible", "multipleRoofSegments", "inverterOrEquipmentVisible"],
    "distinctModuleGridVisible": ["highConfidenceDetection", "significantPanelAreaCoverage", "clearRoofView"],
    "significantPanelAreaCoverage": ["largeCommercialInstallation", "distinctModuleGridVisible"],
    "possibleFalsePositiveRoofLikePattern": ["lowConfidenceNeedsVerification", "obscuredOrUnclearView"],
    "shadowsMayCauseMissedModules": ["partialOcclusionDetected", "lowConfidenceNeedsVerification"],
    "partialOcclusionDetected": ["nearbyObjectsObstructing", "lowResolutionImage"],
    "lowResolutionImage": ["lowConfidenceNeedsVerification", "partialOcclusionDetected"],
    "glareOrSunReflection": ["lowConfidenceNeedsVerification", "partialOcclusionDetected"],
    "tiltedPanelsDetected": ["groundMountedArrayDetected", "distinctModuleGridVisible"],
    "groundMountedArrayDetected": ["largeCommercialInstallation", "trackerSystemsDetected", "bifacialPanelsDetected"],
    "trackerSystemsDetected": ["groundMountedArrayDetected", "bifacialPanelsDetected"],
    "bifacialPanelsDetected": ["groundMountedArrayDetected", "trackerSystemsDetected"],
    "nearbyObjectsObstructing": ["partialOcclusionDetected", "obscuredOrUnclearView"],
    "possibleDebrisOrDamage": ["partialOcclusionDetected", "nearbyObjectsObstructing"],
    "multipleRoofSegments": ["residentialSystemDetected", "distinctModuleGridVisible"],
    "inconsistentModuleSpacing": ["distinctModuleGridVisible", "possibleDebrisOrDamage"],
    "inverterOrEquipmentVisible": ["residentialSystemDetected", "recentInstallationLikely"],
    "recentInstallationLikely": ["highConfidenceDetection", "inverterOrEquipmentVisible"],
    "historicalChangeDetected": ["recentInstallationLikely", "possibleDebrisOrDamage"]
}

# Helper functions (feature builders)
def clamp(x, a, b): return max(a, min(b, x))

def panels_to_area(panel_count):
    return panel_count * random.uniform(1.5, 2.05)

def area_to_capacity_kw(area_sqm):
    return area_sqm * 0.18 * 0.85

def sample_resolution_for_label(label_key):
    # Give resolutions that support the label when needed
    if label_key in ("distinctModuleGridVisible", "highConfidenceDetection", "recentInstallationLikely"):
        return random.uniform(50, 140)
    if label_key in ("lowResolutionImage", "possibleFalsePositiveRoofLikePattern"):
        return random.uniform(8, 24)
    if label_key in ("partialOcclusionDetected", "nearbyObjectsObstructing", "glareOrSunReflection"):
        return random.uniform(18, 60)
    if label_key in ("largeCommercialInstallation", "groundMountedArrayDetected", "trackerSystemsDetected"):
        return random.uniform(30, 100)
    return random.uniform(24, 90)

def derive_confidence(panel_count, pix_per_panel, occluded=False, glare=False, low_res=False):
    if panel_count == 0:
        base_conf = random.uniform(0.12, 0.45)
    else:
        quality = clamp((pix_per_panel - 8) / 190.0, 0.0, 1.0)
        base_conf = clamp(0.35 + 0.65 * quality * min(1.0, (panel_count ** 0.5) / 15.0), 0.02, 0.99)
    if glare: base_conf -= random.uniform(0.03, 0.18)
    if occluded: base_conf -= random.uniform(0.08, 0.3)
    if low_res: base_conf -= random.uniform(0.05, 0.25)
    return clamp(base_conf + random.uniform(-0.03, 0.03), 0.01, 0.995)

# Build a single sample biased for target label
def build_sample_for_target(target_label):
    # Start with a base 'site' depending on typical target semantics
    kind = random.choice(["residential","commercial","ground","industrial"])
    # rough base panel choices
    if target_label in ("largeCommercialInstallation","groundMountedArrayDetected","trackerSystemsDetected","bifacialPanelsDetected"):
        base_panels = random.choice([40, 60, 90, 120])
    elif target_label in ("residentialSystemDetected","multipleRoofSegments"):
        base_panels = random.choice([4, 6, 8, 12])
    elif target_label in ("possibleFalsePositiveRoofLikePattern",):
        base_panels = 0
    else:
        base_panels = random.choice([2,4,6,8,12,20,30])

    panel_count = int(max(0, round(base_panels * random.uniform(0.85, 1.2))))
    # force zero for false positive label
    if target_label == "possibleFalsePositiveRoofLikePattern":
        panel_count = 0

    pix = sample_resolution_for_label(target_label)
    low_res = pix < 24
    # occlusion & glare logic
    occluded = random.random() < 0.18 if target_label in ("partialOcclusionDetected","nearbyObjectsObstructing") else random.random() < 0.08
    glare = random.random() < 0.08 if target_label == "glareOrSunReflection" else False

    area = panels_to_area(panel_count)
    capacity = area_to_capacity_kw(area)
    confidence = derive_confidence(panel_count, pix, occluded=occluded, glare=glare, low_res=low_res)

    # Start with primary note
    notes_internal = [target_label]

    # Add 2-4 compatible co-labels (no contradictions)
    possible = COMPAT.get(target_label, [])
    # Shuffle and pick a few, but ensure we don't pick contradictory ones
    random.shuffle(possible)
    num_extra = random.randint(2, 4)
    for cand in possible:
        if len(notes_internal) >= 1 + num_extra:
            break
        # avoid direct contradictions explicitly
        if (target_label == "highConfidenceDetection" and cand == "lowConfidenceNeedsVerification"): continue
        if (target_label == "lowConfidenceNeedsVerification" and cand == "highConfidenceDetection"): continue
        notes_internal.append(cand)

    # If not enough candidates, pick safe extras from neighbors (distinctModuleGridVisible, etc.)
    if len(notes_internal) < 1 + num_extra:
        extras = [k for k in LABEL_KEYS if k not in notes_internal]
        random.shuffle(extras)
        for cand in extras:
            if len(notes_internal) >= 1 + num_extra:
                break
            # very cheap sanity checks
            if (cand == "highConfidenceDetection" and confidence < 0.6): continue
            if (cand == "lowConfidenceNeedsVerification" and confidence > 0.7): continue
            notes_internal.append(cand)

    # final human-readable notes (dedup)
    notes_hr = [DISPLAY[k] for k in dict.fromkeys(notes_internal)]

    sample = {
        "image_path": f"synthetic_{uuid.uuid4().hex}.png",
        "confidence": round(confidence, 4),
        "panel_count": int(panel_count),
        "area_sqm": round(float(area), 3),
        "capacity_kw": round(float(capacity), 3),
        "has_solar": bool(panel_count > 0),
        "qc_notes": notes_hr
    }
    return sample

def generate_balanced_multilabel(n_per_label=40, out=OUT):
    samples = []
    # primary-target generation
    for label in LABEL_KEYS:
        for _ in range(n_per_label):
            s = build_sample_for_target(label)
            samples.append(s)

    # Shuffle
    random.shuffle(samples)

    # Save and print stats
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, indent=2))

    counts = {}
    for s in samples:
        for lab in s["qc_notes"]:
            counts[lab] = counts.get(lab, 0) + 1

    print(f"✔ Generated {len(samples)} samples to {out}")
    print("Support counts (human label -> occurrences):")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    return samples

if __name__ == "__main__":
    # default 40 primary samples per label -> 25*40 = 1000 samples
    generate_balanced_multilabel(n_per_label=40)
