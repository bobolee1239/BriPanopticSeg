# panoptic_module.py
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import List, Dict, Union
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

class PanopticTrainRoutine(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        category_table: List[Dict[str, Union[str, List[int], bool]]] = None,
        visualize_every_n: int = 50
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate = learning_rate
        self.category_table = category_table
        self.visualize_every_n = visualize_every_n

    def forward(self, images: List[torch.Tensor]) -> Dict[str, Union[List, torch.Tensor]]:
        return self.model.forward_infer(images)

    def training_step(self, batch: Dict[str, Union[List[torch.Tensor], List[Dict[str, torch.Tensor]]]], batch_idx: int) -> torch.Tensor:
        images = batch['images']
        targets = batch['targets']

        # Check for any target with no boxes
        if any(t['boxes'].numel() == 0 for t in targets):
            self.log("train/skipped_empty", 1.0, prog_bar=False, on_step=True)
            return None

        loss_dict = self.model.forward_train(images, targets)
        total_loss = sum(loss_dict.values())

        for k, v in loss_dict.items():
            self.log(f'train/{k}', v, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/loss_total', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch: Dict[str, Union[List[torch.Tensor], List[Dict[str, torch.Tensor]]]], batch_idx: int) -> None:
        images = batch['images']
        targets = batch['targets']
        outputs = self.model.forward_infer(images)

        # self.log('val/num_preds', float(sum(len(p['labels']) for p in outputs)), on_step=False, on_epoch=True)

        # import pdb; pdb.set_trace()
        if self.category_table and batch_idx % self.visualize_every_n == 0:
            for i, (image, pred) in enumerate(zip(images, outputs['instances'])):
                vis_img = self._visualize_prediction(image, pred)
                self.logger.experiment.add_image(f'val/panoptic_pred_{batch_idx}_{i}', vis_img, self.global_step)

            sem_seg = outputs['sem_logits'].argmax(dim=1)  # [B, H, W]
            for i in range(len(images)):
                sem_img = self._visualize_semantic_segmentation(images[i], sem_seg[i])
                self.logger.experiment.add_image(f'val/sem_pred_{batch_idx}_{i}', sem_img, self.global_step)
        return

    def _visualize_prediction(self, image: torch.Tensor, prediction: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Convert image to [3, H, W] uint8
        img = (image * 255).to(torch.uint8).cpu()
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)

        masks = (prediction['masks'] > 0.5).squeeze(1).cpu()  # [N, H, W]
        labels = prediction['labels'].cpu().tolist()
        boxes = prediction.get('boxes', torch.empty((0, 4))).cpu()

        # Build colors and label names
        colors = [tuple(self.category_table[lbl].get("color", [255, 255, 255])) for lbl in labels]
        names = [self.category_table[lbl].get("name", str(lbl)) for lbl in labels]
        isthing = [self.category_table[lbl].get("isthing", False) for lbl in labels]

        # Split masks/labels/boxes by thing/stuff
        thing_masks, thing_boxes, thing_labels, thing_colors = [], [], [], []
        stuff_masks, stuff_colors = [], []

        for i, is_thing in enumerate(isthing):
            if is_thing:
                thing_masks.append(masks[i])
                thing_boxes.append(boxes[i])
                thing_labels.append(names[i])
                thing_colors.append(colors[i])
            else:
                stuff_masks.append(masks[i])
                stuff_colors.append(colors[i])

        # Draw stuff (segmentation only)
        if len(stuff_masks) > 0:
            img = draw_segmentation_masks(
                img,
                masks=torch.stack(stuff_masks),
                colors=stuff_colors,
                alpha=0.5
            )

        # Draw things (segmentation + boxes + labels)
        if len(thing_masks) > 0:
            img = draw_segmentation_masks(
                img,
                masks=torch.stack(thing_masks),
                colors=thing_colors,
                alpha=0.5
            )
            img = draw_bounding_boxes(
                img,
                boxes=torch.stack(thing_boxes).int(),
                labels=thing_labels,
                colors=thing_colors,
                font_size=16,
                width=2
            )

        return img.float() / 255.0

    def _visualize_semantic_segmentation(self, image: torch.Tensor, sem_pred: torch.Tensor) -> torch.Tensor:
        """
        image: [3, H, W] float tensor in [0, 1]
        sem_pred: [H', W'] long tensor with class indices
        """
        H, W = image.shape[-2:]
        sem_pred = sem_pred.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H', W']
        sem_pred = F.interpolate(sem_pred, size=(H, W), mode='nearest').squeeze().long()  # [H, W]

        img = (image * 255).to(torch.uint8).cpu()
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)

        sem_pred = sem_pred.cpu()

        # Extract unique classes
        unique_classes = torch.unique(sem_pred)
        masks = [(sem_pred == class_id) for class_id in unique_classes]

        # Pick colors for each class
        colors = [
            tuple(self.category_table[class_id.item()].get("color", [255, 255, 255]))
            for class_id in unique_classes
        ]

        vis = draw_segmentation_masks(
            img,
            masks=torch.stack(masks),
            colors=colors,
            alpha=0.5
        )

        return vis.float() / 255.0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

