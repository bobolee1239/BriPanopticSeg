import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import random

from typing import Dict, Tuple


def build_category_table(category_list: list[dict]) -> Dict[int, Dict]:
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


def visualize_panoptic_target(
    image_path: str,
    target: Dict,
    category_table: Dict[int, Dict],
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (16, 10)
) -> Tuple[plt.Figure, plt.Axes]:
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
import random

from build_maskrcnn_target import build_maskrcnn_target

setname = 'val'
d_panoptic_imgs = f'Dataset/cityscapes/gtFine/cityscapes_panoptic_{setname}'
d_imgs = os.path.join('Dataset/cityscapes/leftImg8bit', setname)

f_json = os.path.join('./Dataset/cityscapes/gtFine',
                      f'cityscapes_panoptic_{setname}.json',
                      )

# Load entry from panoptic_train.json
with open(f_json) as f:
    panoptic_json = json.load(f)


category_table = build_category_table(panoptic_json["categories"])
datas = random.choices(panoptic_json['annotations'], k=3)

for data in datas:
    img_id = data['image_id']
    print(f'[I] {img_id}')

    png_path = os.path.join(d_panoptic_imgs, f'{img_id}_gtFine_panoptic.png')

    case = img_id.split('_')[0]
    img_path = os.path.join(d_imgs, case, f'{img_id}_leftImg8bit.png')
    target = build_maskrcnn_target(png_path, data["segments_info"])

    # import pdb; pdb.set_trace()
    fig, ax = visualize_panoptic_target(img_path, 
                                        target, 
                                        category_table,
                                        alpha=0.7)

plt.show()
