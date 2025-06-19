# train.py
import os
import json
import torch
import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from BriPanopticSeg.Model.PanopticFpn import PanopticFPN
from BriPanopticSeg.Train.PanopticTrainRoutine import PanopticTrainRoutine
from BriPanopticSeg.Data.CityscapesPanopticDataset import CityscapesPanopticDataset
from BriPanopticSeg.Data.PanopticDataModule import PanopticDataModule

from typing import Dict, List, Union, Any

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


def create_dataset(setname: str,
                   category_table: Dict[int, Dict[str, Any]],
                   ) -> torch.utils.data.Dataset:
    d_panoptic_imgs = f'Dataset/cityscapes/gtFine/cityscapes_panoptic_{setname}'
    d_imgs = os.path.join('Dataset/cityscapes/leftImg8bit', setname)
    f_json = os.path.join('./Dataset/cityscapes/gtFine', f'cityscapes_panoptic_{setname}.json')
    with open(f_json) as f:
        panoptic_json = json.load(f)
    annotations = panoptic_json['annotations']

    transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                ],
                )
    cropFcn = A.RandomCrop(height=512, width=1024)
    return CityscapesPanopticDataset(
                root_img_dir=d_imgs,
                root_panoptic_dir=d_panoptic_imgs,
                panoptic_annotations=annotations,
                transform=transform,
                cropFcn=cropFcn,
                category_table=category_table,
                )

def create_datamodule(batchsize: int,
                      num_workers: int,
                      category_table: Dict[int, Dict[str, Any]],
                      ) -> pl.LightningDataModule:
    return PanopticDataModule(train_dataset=create_dataset('train', category_table),
                              val_dataset=create_dataset('val', category_table),
                              batch_size=batchsize,
                              num_workers=num_workers
                              )

def main() -> int:
    # === Configs ===
    setname = 'train'
    f_json = os.path.join('./Dataset/cityscapes/gtFine', f'cityscapes_panoptic_{setname}.json')
    with open(f_json) as f:
        panoptic_json = json.load(f)

    annotations = panoptic_json['annotations']
    category_table = build_category_table(panoptic_json['categories'])

    num_things = sum(1 for id, info in category_table.items() if info.get("isthing", 1) == 1)
    num_stuff = sum(1 for id, info in category_table.items() if info.get("isthing", 1) == 0)
    print(f'[I] #things={num_things}, #stuff={num_stuff}')
    # === Model and Modules ===
    model = PanopticFPN(num_classes=num_things+num_stuff,
                        )
    datamodule = create_datamodule(batchsize=16,
                                   num_workers=8,
                                   category_table=category_table,
                                   )
    
    lr = 1e-4
    train_routine = PanopticTrainRoutine(
                        model=model,
                        learning_rate=lr,
                        category_table=panoptic_json['categories'],
                        visualize_every_n=50
                        )
    # === Logging & Checkpoints ===
    logger = TensorBoardLogger("logs", 
                               name="panoptic_training",
                               )
    checkpoint_cb: ModelCheckpoint = ModelCheckpoint(
        monitor="train/loss_total",
        mode="min",
        save_top_k=3,
        dirpath="checkpoints",
        filename="panoptic-fpn-{epoch:02d}-{train/loss_total:.4f}"
    )
    lr_monitor_cb: LearningRateMonitor = LearningRateMonitor(logging_interval='step')

    # === Trainer ===
    trainer: pl.Trainer = pl.Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor_cb],
        log_every_n_steps=30,
        accelerator="gpu",
        devices=1
    )
    # === Run Training ===
    trainer.fit(train_routine, 
                datamodule=datamodule,
                )
    return 0

if __name__ == "__main__":
    # import argparse
    main()

