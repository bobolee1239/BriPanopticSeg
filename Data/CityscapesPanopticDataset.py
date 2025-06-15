import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, List, Tuple, Dict, Optional, Any

from build_maskrcnn_target import build_maskrcnn_target


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
