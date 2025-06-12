import numpy as np
from PIL import Image
import torch

def build_maskrcnn_target(png_path, segments_info):
    """
    Cityscapes version — uses int32 grayscale PNG where pixel = segment_id.
    """
    # Read as raw 32-bit image (mode 'I' = int32)
    segment_ids = np.array(Image.open(png_path).convert("I"))
    H, W = segment_ids.shape

    masks = []
    boxes = []
    labels = []

    for segment in segments_info:
        seg_id = segment["id"]
        category_id = segment["category_id"]

        mask = (segment_ids == seg_id).astype(np.uint8)
        if mask.sum() == 0:
            continue

        masks.append(mask)
        labels.append(category_id)

        # Compute bounding box from mask
        y_indices, x_indices = np.where(mask)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        boxes.append([x_min, y_min, x_max, y_max])

    if len(masks) == 0:
        return None

    target = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8)
    }

    return target

# ---

import os
import json

setname = 'val'
f_json = os.path.join('./Dataset/cityscapes/gtFine',
                      f'cityscapes_panoptic_{setname}.json',
                      )
d_imgs = os.path.join('Dataset/cityscapes/gtFine',
                      f'cityscapes_panoptic_{setname}'
                      )
with open(f_json) as f:
    panoptic_json = json.load(f)

entry = panoptic_json["annotations"][0]
print(f'[D] entry: {entry}')
target = build_maskrcnn_target(os.path.join(d_imgs, entry["file_name"]),
                               entry["segments_info"],
                               )

print(target.keys())  # ['boxes', 'labels', 'masks']
print(target)
