import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import Dict, Any, List, Tuple

from BriPanopticSeg.Data.CityscapesPanopticDataset import CityscapesPanopticDataset
from BriPanopticSeg.Data.PanopticDataModule import PanopticDataModule


def build_category_table(categories: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {
        cat['id']: {
            'name': cat['name'],
            'color': tuple(cat['color']),
            'isthing': cat.get('isthing', 1)
        }
        for cat in categories
    }

def visualize_sample(
    image_tensor: Any,
    target: Dict[str, Any],
    category_table: Dict[int, Dict[str, Any]],
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

        cat_info = category_table.get(label_id, {
            'name': f'id:{label_id}',
            'color': (255, 255, 255),
            'isthing': 1
        })

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

def test_cityscapes_datamodule() -> None:
    setname = 'train'
    d_panoptic_imgs = f'Dataset/cityscapes/gtFine/cityscapes_panoptic_{setname}'
    d_imgs = os.path.join('Dataset/cityscapes/leftImg8bit', setname)

    f_json = os.path.join('./Dataset/cityscapes/gtFine', f'cityscapes_panoptic_{setname}.json')

    with open(f_json) as f:
        panoptic_json = json.load(f)

    annotations = panoptic_json['annotations']
    category_table = build_category_table(panoptic_json['categories'])

    dataset = CityscapesPanopticDataset(
        root_img_dir=d_imgs,
        root_panoptic_dir=d_panoptic_imgs,
        panoptic_annotations=annotations,
        transform=None
    )

    datamodule = PanopticDataModule(train_dataset=dataset,
                                    val_dataset=None,
                                    batch_size=2,
                                    )

    dataloader = datamodule.train_dataloader()

    data = next(iter(dataloader))
    imgs = data['images']
    targets = data['targets']

    N = len(imgs)
    for idx in range(N):
        img_tensor = imgs[idx]
        target = targets[idx]
        visualize_sample(img_tensor, 
                         target, 
                         category_table, 
                         f'sample[{idx}]')

    plt.show()
    return 

# ---

if __name__ == '__main__':
    test_cityscapes_datamodule()
