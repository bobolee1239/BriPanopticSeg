# panoptic_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union
import pytorch_lightning as pl

class PanopticTrainRoutine(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, images: List[torch.Tensor]) -> Dict[str, Union[List, torch.Tensor]]:
        return self.model.forward_infer(images)

    def training_step(self, 
                      batch: Dict[str, Union[List[torch.Tensor], List[Dict[str, torch.Tensor]]]], 
                      batch_idx: int,
                      ) -> torch.Tensor:
        images = batch['images']
        targets = batch['targets']
        loss_dict = self.model.forward_train(images, targets)
        total_loss = sum(loss_dict.values())

        for k, v in loss_dict.items():
            self.log(k, v, prog_bar=True)
        self.log('loss_total', total_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

