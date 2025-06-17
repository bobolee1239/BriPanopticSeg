# panoptic_datamodule.py
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict, List


class PanopticDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 2,
        num_workers: int = 4
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        images, targets = zip(*batch)
        return {
            'images': list(images),
            'targets': list(targets)
        }

