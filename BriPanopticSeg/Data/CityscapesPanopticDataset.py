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
        transform: Optional[Callable] = None,
    ) -> None:
        self._root_img_dir = root_img_dir
        self._root_panoptic_dir = root_panoptic_dir
        self._panoptic_annotations = panoptic_annotations
        self._transform = transform
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
                                                        )
        if self._transform:
            image_np = np.array(image)
            masks = target['masks'].numpy()
            boxes = target['boxes'].numpy()
            labels = target['labels'].tolist()
            result: Dict[str, Any] = self._transform(image=image_np,
                                                     masks=masks,
                                                     bboxes=boxes,
                                                     category_ids=labels,
                                                     )
            image_tensor = (
                torch.from_numpy(result['image']).permute(2, 0, 1).float() 
                    / 255.0
                )
            target = {
                'masks': torch.as_tensor(result['masks'], dtype=torch.uint8),
                'boxes': torch.as_tensor(result['bboxes'], dtype=torch.float32),
                'labels': torch.as_tensor(result['category_ids'], dtype=torch.int64)
            }
        else:
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image_tensor, target
