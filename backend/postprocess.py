# postprocess.py
import numpy as np
from math import cos, radians
from sklearn.cluster import DBSCAN

def meters_per_pixel(lat, zoom):
    return 156543.03392 * cos(radians(lat)) / (2 ** zoom)

def aggregate_predictions(dets, w, h, lat, zoom):
    boxes = [d["box"] for d in dets]
    confs = [d["confidence"] for d in dets]
    masks = [d["mask"] for d in dets if d["mask"] is not None]

    mpp = meters_per_pixel(lat, zoom)

    # area calc (bbox-based)
    total_pixels = 0.0
    for (x1,y1,x2,y2) in boxes:
        total_pixels += max(0,(x2-x1)) * max(0,(y2-y1))

    area_sqm = total_pixels * (mpp ** 2)
    capacity_kw = (area_sqm * 180.0 / 1000.0) * 0.75

    # clustering centroids
    if boxes:
        centers = np.array([[(x1+x2)/2, (y1+y2)/2] for (x1,y1,x2,y2) in boxes])
        eps_px = 10.0 / mpp
        db = DBSCAN(eps=eps_px, min_samples=1).fit(centers)
        clusters = []
        for lbl in np.unique(db.labels_):
            idxs = np.where(db.labels_ == lbl)[0]
            cx = float(np.mean(centers[idxs,0]))
            cy = float(np.mean(centers[idxs,1]))
            clusters.append({
                "cluster_id": int(lbl),
                "num_detections": len(idxs),
                "mean_confidence": float(np.mean([confs[i] for i in idxs])),
                "centroid_px": [cx,cy]
            })
    else:
        clusters = []

    return {
        "has_solar": area_sqm > 0.1,
        "panel_count": int(max(1, round(area_sqm / 1.7))) if area_sqm > 0 else 0,
        "area_sqm": area_sqm,
        "capacity_kw": capacity_kw,
        "confidence": float(np.mean(confs)) if confs else 0.0,
        "boxes": boxes,
        "masks": masks,
        "clusters": clusters,
        "mpp": mpp
    }
