import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import random

def build_category_table(category_list):
    """
    Convert `categories` from panoptic JSON into a lookup dict:
    {category_id: {"name", "color", "isthing"}}
    """
    return {
        cat["id"]: {
            "name": cat["name"],
            "color": tuple(cat["color"]),
            "isthing": cat.get("isthing", 1)  # default to 1 just in case
        }
        for cat in category_list
    }


def visualize_panoptic_target(image_path, target, category_table, alpha=0.5, figsize=(16, 10)):
    """
    Visualize an image with panoptic segmentation target.

    Args:
        image_path (str): Path to RGB image
        target (dict): Contains "boxes", "masks", "labels" (category_id)
        category_table (dict): {category_id: {"name", "color", "isthing"}}
        alpha (float): Transparency for thing-class masks
        figsize (tuple): Matplotlib figure size

    Returns:
        fig, ax: Matplotlib figure and axis
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    boxes = target["boxes"]
    masks = target["masks"]
    labels = target["labels"]

    H, W, _ = img.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)

    fig, ax = plt.subplots(1, figsize=figsize)

    for i in range(len(boxes)):
        mask = masks[i].numpy()
        cat_id = int(labels[i].item())

        # Lookup category info
        cat_info = category_table.get(cat_id, {
            "name": f"id:{cat_id}",
            "color": (255, 255, 255),
            "isthing": True
        })
        name = cat_info["name"]
        color = [c / 255.0 for c in cat_info["color"]]
        isthing = cat_info["isthing"]

        # Less transparent for stuff
        mask_alpha = alpha if isthing else (min(alpha*1.99, 0.98))

        # Overlay mask
        for c in range(3):
            overlay[:, :, c] += mask * color[c] * mask_alpha

        # Get bounding box even for stuff, but don't draw it
        box = boxes[i].numpy()
        x1, y1, x2, y2 = map(round, box)
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)

        # Draw box only for "things"
        if isthing:
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

        # Always draw text label
        ax.text(
            x1, y1 - 5, name,
            color=color, fontsize=10, backgroundcolor='white'
        )

    # Blend and display
    overlay = np.clip(overlay, 0, 1)
    blended = (img / 255.0) * (1 - alpha) + overlay * alpha

    ax.imshow(blended)
    ax.axis("off")
    ax.set_title("Panoptic Target Visualization (things + stuff)")
    return fig, ax

# ---

import os
import json


from build_maskrcnn_target import build_maskrcnn_target

# Example paths
img_path = 'Dataset/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_010156_leftImg8bit.png'
png_path = "Dataset/cityscapes/gtFine/cityscapes_panoptic_val/frankfurt_000001_010156_gtFine_panoptic.png"

setname = 'val'
f_json = os.path.join('./Dataset/cityscapes/gtFine',
                      f'cityscapes_panoptic_{setname}.json',
                      )

# Load entry from panoptic_train.json
with open(f_json) as f:
    panoptic_json = json.load(f)


entry = None
for idx, e in enumerate(panoptic_json['annotations']):
    if e['file_name'] == os.path.basename(png_path):
        print(f'[D] idx(entry) = {idx}')
        entry = e
assert(entry is not None)

target = build_maskrcnn_target(png_path, entry["segments_info"])

segment_ids = np.array(Image.open(png_path).convert("I"))
import pdb; pdb.set_trace()
category_table = build_category_table(panoptic_json["categories"])
fig, ax = visualize_panoptic_target(img_path, target, category_table)

plt.show()
