import os
from ultralytics import YOLO

verifier = YOLO("verifier1.pt") # loading the model

imgs = os.listdir(os.path.join(".", "eg_test_images"))

paths = []
for img in imgs:
    paths.append(os.path.join(".", "eg_test_images", img))

results = verifier.predict(paths)

count = 0
for result in results:
    if result.masks:
        xy = result.masks.xy # masks in polygon format
        xyn = result.masks.xyn # normalized
        masks = result.masks.data # mask in matrix format (num objects x H x W)
    result.save(os.path.join(".", "eg_test_output", f"result{count}.png"))
    count += 1