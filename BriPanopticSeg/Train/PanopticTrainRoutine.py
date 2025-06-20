# panoptic_module.py
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch_optimizer as optim

from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import nms
from typing import List, Dict, Union

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
        with torch.no_grad():
            outputs = self.model.forward_infer(images)
        if self.category_table and batch_idx % self.visualize_every_n == 0:
            sem_seg = outputs['sem_logits'].argmax(dim=1)  # [B, H, W]
            for i, (image, pred) in enumerate(zip(images, outputs['instances'])):
                vis_img = self._visualize_prediction(image, pred)
                self.logger.experiment.add_image(f'val/instance_pred_{batch_idx}_{i}', vis_img, self.global_step)

                sem_img = self._visualize_semantic_segmentation(image, sem_seg[i])
                self.logger.experiment.add_image(f'val/sem_pred_{batch_idx}_{i}', sem_img, self.global_step)

                fused_img = self._fuse_panoptic_prediction(image, pred, sem_seg[i])
                self.logger.experiment.add_image(f'val/panoptic_{batch_idx}_{i}', fused_img, self.global_step)
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

    def _fuse_panoptic_prediction(self, image: torch.Tensor, instance_pred: Dict[str, torch.Tensor], sem_pred: torch.Tensor) -> torch.Tensor:
        img = (image * 255).to(torch.uint8).cpu()
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)

        sem_pred = sem_pred.cpu()
        instance_masks = (instance_pred['masks'] > 0.5).squeeze(1).cpu()
        instance_boxes = instance_pred['boxes'].cpu()
        instance_scores = instance_pred['scores'].cpu()
        instance_labels = instance_pred['labels'].cpu()

        keep = nms(instance_boxes, instance_scores, iou_threshold=0.5)
        instance_masks = instance_masks[keep]
        instance_boxes = instance_boxes[keep]
        instance_labels = instance_labels[keep]

        sem_unique = torch.unique(sem_pred)
        stuff_masks, stuff_boxes, stuff_labels, stuff_colors = [], [], [], []

        for class_id in sem_unique:
            class_id_int = class_id.item()
            mask = sem_pred == class_id
            if mask.sum() == 0:
                continue

            # Skip if a thing class already predicted in instance head
            if any((lbl.item() == class_id_int) and self.category_table[lbl.item()].get("isthing", False) for lbl in instance_labels):
                continue
            stuff_masks.append(mask)
            stuff_labels.append(self.category_table[class_id_int].get("name", str(class_id_int)))
            stuff_colors.append(tuple(self.category_table[class_id_int].get("color", [255, 255, 255])))

            y_indices, x_indices = torch.where(mask)
            if len(x_indices) > 0 and len(y_indices) > 0:
                cx = int(x_indices.float().mean())
                cy = int(y_indices.float().mean())
                box = torch.tensor([cx - 1, cy - 1, cx + 1, cy + 1])
            else:
                box = torch.tensor([0, 0, 0, 0])
            stuff_boxes.append(box)

        if stuff_masks:
            img = draw_segmentation_masks(
                img,
                masks=torch.stack(stuff_masks),
                colors=stuff_colors,
                alpha=0.5
            )
            img = draw_bounding_boxes(
                img,
                boxes=torch.stack(stuff_boxes).int(),
                labels=stuff_labels,
                colors=stuff_colors,
                font_size=16,
                width=1
            )

        thing_indices = [
            i for i, lbl in enumerate(instance_labels)
            if self.category_table[lbl.item()].get("isthing", False)
        ]

        if thing_indices:
            instance_masks = instance_masks[thing_indices]
            instance_boxes = instance_boxes[thing_indices]
            instance_labels = instance_labels[thing_indices]

            inst_colors = [tuple(self.category_table[lbl.item()].get("color", [255, 255, 255])) for lbl in instance_labels]
            inst_labels = [self.category_table[lbl.item()].get("name", str(lbl.item())) for lbl in instance_labels]

            img = draw_segmentation_masks(
                img,
                masks=instance_masks,
                colors=inst_colors,
                alpha=0.6
            )
            img = draw_bounding_boxes(
                img,
                boxes=instance_boxes.int(),
                labels=inst_labels,
                colors=inst_colors,
                font_size=16,
                width=2
            )

        return img.float() / 255.0

    def configure_optimizers(self):
        base_optim = torch.optim.AdamW(
                        self.parameters(), 
                        lr=self.learning_rate, 
                        weight_decay=1e-4,
                        )
        # optimizer = optim.Lookahead(base_optim)
        optimizer = base_optim
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, 
                        T_max=50,
                        )
        return [optimizer], [scheduler]
