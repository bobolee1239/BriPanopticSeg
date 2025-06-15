import torch
import numpy as np
from PIL import Image

def rgb2id(color):
    """
    Convert RGB-encoded panoptic image (uint8) to COCO-style segment IDs:
    id = R + 256*G + 256*256*B
    """
    return (color[:, :, 0].astype(np.int32)
            + color[:, :, 1].astype(np.int32) * 256
            + color[:, :, 2].astype(np.int32) * 256 * 256)

def build_maskrcnn_target(png_path, segments_info):
    """
    Build a Mask R-CNN style target dict from a COCO-formatted panoptic PNG + segments_info JSON.

    Args:
        png_path (str): Path to COCO-style panoptic PNG (RGB).
        segments_info (list[dict]): List of segment metadata dicts.

    Returns:
        dict or None: {"boxes":[N,4],"labels":[N],"masks":[N,H,W]} or None if no segments.
    """
    pan_png = np.array(Image.open(png_path).convert("RGB"))
    segment_ids = rgb2id(pan_png)
    H, W = segment_ids.shape

    boxes, labels, masks = [], [], []

    for segment in segments_info:
        sid = segment["id"]
        cat = segment["category_id"]

        mask = (segment_ids == sid).astype(np.uint8)
        if mask.sum() == 0:
            continue

        labels.append(cat)
        masks.append(mask)

        ys, xs = np.where(mask)
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])

    if not masks:
        return None

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "masks": torch.tensor(np.stack(masks), dtype=torch.uint8),
    }

# ---

if __name__ == '__main__':
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
