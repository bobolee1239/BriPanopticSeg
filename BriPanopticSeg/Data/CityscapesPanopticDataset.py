import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, List, Tuple, Dict, Optional, Any

def rgb2id(color):
    """
    Convert RGB-encoded panoptic image (uint8) to COCO-style segment IDs:
    id = R + 256*G + 256*256*B
    """
    return (color[:, :, 0].astype(np.int32)
            + color[:, :, 1].astype(np.int32) * 256
            + color[:, :, 2].astype(np.int32) * 256 * 256)

def build_maskrcnn_target(png_path, 
                          segments_info,
                          category_table,):
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

        labels.append(category_table[cat]['trainId'])
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


def build_sem_seg_from_masks(masks, labels, height, width):
    sem_seg = torch.zeros((height, width), dtype=torch.long)

    for mask, label in zip(masks, labels):
        sem_seg[mask.bool()] = label

    return sem_seg

def extract_bounding_box(mask: np.ndarray) -> np.ndarray:
    """Extract the bounding box of a mask.

    :param mask: HxW numpy array
    :return: bounding box
    """
    pos = np.where(mask)  # TODO: Check if np.nonzero can be used instead

    if not (pos[0].size or pos[1].size):
        return np.array([0, 0, 0, 0])

    xmin = np.min(pos[1])
    xmax = np.max(pos[1]) + 1
    ymin = np.min(pos[0])
    ymax = np.max(pos[0]) + 1
    return np.array([xmin, ymin, xmax, ymax])

class CityscapesPanopticDataset(Dataset):
    """
    PyTorch dataset for Cityscapes panoptic segmentation.

    Args:
        root_img_dir (str): Path to Cityscapes leftImg8bit/<split>
        root_panoptic_dir (str): Path to corresponding panoptic PNGs
        panoptic_annotations (List[Dict]): List of annotation entries from JSON
        transform (Optional[Callable]): Albumentations-style transform:
            image, masks, bboxes, category_ids → image, masks, bboxes, category_ids
    """

    def __init__(
        self,
        root_img_dir: str,
        root_panoptic_dir: str,
        panoptic_annotations: List[Dict[str, Any]],
        category_table      : Dict[int, Dict[str, Any]],
        transform: Optional[Callable] = None,
        cropFcn             : Optional[Callable]=None,
    ) -> None:
        self._root_img_dir = root_img_dir
        self._root_panoptic_dir = root_panoptic_dir
        self._panoptic_annotations = panoptic_annotations
        self._transform = transform

        self._category_table = category_table
        self._cropFcn = cropFcn
        return

    def __len__(self) -> int:
        return len(self._panoptic_annotations)

    def __getitem__(self, 
                    idx: int,
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        '''
        '''
        annotation: Dict[str, Any] = self._panoptic_annotations[idx]
        image_id: str = annotation['image_id']
        city: str = image_id.split('_')[0]

        f_img: str = os.path.join(self._root_img_dir, 
                                  city, 
                                  f'{image_id}_leftImg8bit.png',
                                  )
        f_png: str = os.path.join(self._root_panoptic_dir, 
                                  f'{image_id}_gtFine_panoptic.png',
                                  )
        image: Image.Image = Image.open(f_img).convert('RGB')
        target: Optional[Dict[str, torch.Tensor]] = build_maskrcnn_target(
                                                        f_png,
                                                        annotation['segments_info'],
                                                        self._category_table,
                                                        )
        if self._transform:
            image_np = np.array(image)
            masks = target['masks'].numpy()
            boxes = target['boxes'].numpy()
            labels = target['labels'].tolist()
            result = self._transform(image=image_np,
                                                     masks=masks,
                                                     )
            result = self._cropFcn(image=result['image'],
                                   masks=result['masks'],
                                   )
            keepit = [np.any(m) for m in result['masks']]
            masks = [result['masks'][n] for n in range(len(keepit)) if keepit[n]]
            boxes = [extract_bounding_box(mask) for mask in masks]
            labels = [target['labels'][n] for n in range(len(keepit)) if keepit[n]]

            image_tensor = (
                torch.from_numpy(result['image']).permute(2, 0, 1).float() 
                    / 255.0
                )
            target = {
                'masks': torch.as_tensor(np.array(masks), dtype=torch.uint8),
                'boxes': torch.as_tensor(np.array(boxes), dtype=torch.float32),
                'labels': torch.as_tensor(np.array(labels), dtype=torch.int64)
            }
        else:
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        height, width = image_tensor.shape[1:]
        sem_seg = build_sem_seg_from_masks(target['masks'], 
                                           target['labels'], 
                                           height,
                                           width,
                                           )
        target['sem_seg'] = sem_seg
        return image_tensor, target
