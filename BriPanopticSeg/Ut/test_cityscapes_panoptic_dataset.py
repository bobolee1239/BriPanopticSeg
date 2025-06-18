import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import Dict, Any, List, Tuple

import albumentations as A
from BriPanopticSeg.Data.CityscapesPanopticDataset import CityscapesPanopticDataset

def build_category_table(categories: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {
        cat['id']: {
            'name': cat['name'],
            'color': tuple(cat['color']),
            'isthing': cat.get('isthing', 1),
            'trainId': trainId,
        }
        for trainId, cat in enumerate(categories)
    }

def visualize_sample(
    image_tensor: Any,
    target: Dict[str, Any],
    category_table: List[Dict[str, Any]],
    sample_idx: int
) -> None:
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    masks = target['masks'].numpy()
    labels = target['labels'].numpy()
    boxes = target['boxes'].numpy()

    H, W, _ = image_np.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)

    fig, ax = plt.subplots(1, figsize=(14, 8))

    for i in range(len(masks)):
        mask = masks[i]
        label_id = int(labels[i])
        box = boxes[i].astype(int)

        cat_info = category_table[label_id]

        name = cat_info['name']
        color = [c / 255.0 for c in cat_info['color']]
        isthing = cat_info['isthing']

        for c in range(3):
            overlay[:, :, c] += mask * color[c] * 255

        if isthing:
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            text_x, text_y = x1, max(y1 - 5, 5)
        else:
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                text_x, text_y = int(xs.mean()), int(ys.mean())
            else:
                text_x, text_y = 10, 10 + i * 15

        ax.text(
            text_x, text_y, name,
            color=color, fontsize=10, backgroundcolor='white'
        )

    alpha = 0.8
    blended = ((1.0 - alpha) * image_np + alpha * overlay).astype(np.uint8)
    ax.imshow(blended)
    ax.set_title(f'Sample {sample_idx}')
    ax.axis('off')
    return

def test_cityscapes_dataset() -> None:
    setname = 'train'
    d_panoptic_imgs = f'Dataset/cityscapes/gtFine/cityscapes_panoptic_{setname}'
    d_imgs = os.path.join('Dataset/cityscapes/leftImg8bit', setname)

    f_json = os.path.join('./Dataset/cityscapes/gtFine', f'cityscapes_panoptic_{setname}.json')

    with open(f_json) as f:
        panoptic_json = json.load(f)

    annotations = panoptic_json['annotations']
    category_table = panoptic_json['categories']

    transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                ],
                )
    cropFcn = A.RandomCrop(height=512, width=1024)
    dataset = CityscapesPanopticDataset(
        root_img_dir=d_imgs,
        root_panoptic_dir=d_panoptic_imgs,
        panoptic_annotations=annotations,
        category_table=build_category_table(category_table),
        transform=transform,
        cropFcn=cropFcn,
    )

    indices = random.sample(range(len(dataset)), 3)
    for idx in indices:
        img_tensor, target = dataset[idx]
        visualize_sample(img_tensor, 
                         target, 
                         category_table, 
                         f'sample[{idx}]')

    plt.show()
    return 

# ---

if __name__ == '__main__':
    test_cityscapes_dataset()
