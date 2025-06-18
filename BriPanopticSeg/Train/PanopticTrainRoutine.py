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
from torchvision.utils import draw_segmentation_masks

class PanopticTrainRoutine(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        category_table: Dict[int, Dict[str, Union[str, List[int], bool]]] = None,
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

    def _visualize_prediction(self, image: torch.Tensor, prediction: Dict[str, torch.Tensor]) -> torch.Tensor:
        # image: [3, H, W], normalized float tensor in [0, 1]
        img = (image * 255).byte().cpu()
        masks = prediction['masks'] > 0.5
        labels = prediction['labels'].cpu().tolist()
        boxes = prediction['boxes'].cpu() if 'boxes' in prediction else None

        colors = [tuple(self.category_table.get(lbl, {}).get("color", [255, 255, 255])) for lbl in labels]
        mask_tensor = masks.squeeze(1).cpu()

        vis = draw_segmentation_masks(img, mask_tensor, alpha=0.6, colors=colors)
        vis = to_pil_image(vis)

        # Draw bounding boxes and labels
        fig, ax = plt.subplots(1)
        ax.imshow(vis)
        for i, label in enumerate(labels):
            name = self.category_table.get(label, {}).get("name", f"id:{label}")
            color = tuple(np.array(self.category_table.get(label, {}).get("color", [255, 255, 255])) / 255.0)

            if boxes is not None:
                x1, y1, x2, y2 = boxes[i].tolist()
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, name, fontsize=8, color=color, backgroundcolor='white')

        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = plt.imread(buf, format='png')
        plt.close(fig)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        return img_tensor

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

