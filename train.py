import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes_panoptic import register_all_cityscapes_panoptic
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo

# === 1. Dataset Registration ===
CITYSCAPES_ROOT = "./Dataset"

# Clean previous registrations (optional for fresh runs)
for split in ["train", "val"]:
    base = f"cityscapes_fine_panoptic_{split}"
    if base in DatasetCatalog:
        DatasetCatalog.remove(base)
        MetadataCatalog.remove(base)
    for suffix in ["_separated", "_stuffonly"]:
        full = base + suffix
        if full in DatasetCatalog:
            DatasetCatalog.remove(full)
            MetadataCatalog.remove(full)

register_all_cityscapes_panoptic(CITYSCAPES_ROOT)

# === 2. Config Setup ===
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
)

# Dataset
cfg.DATASETS.TRAIN = ("cityscapes_fine_panoptic_train",)
cfg.DATASETS.TEST = ("cityscapes_fine_panoptic_val",)
cfg.DATALOADER.NUM_WORKERS = 4

# Input resolution
cfg.INPUT.MIN_SIZE_TRAIN = (512, 768)
cfg.INPUT.MIN_SIZE_TEST = 512

# === Make it truly from scratch ===
cfg.MODEL.WEIGHTS = ""  # No checkpoint loading
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
cfg.MODEL.RESNETS.DEPTH = 50
cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
cfg.MODEL.RESNETS.NORM = "BN"

# === Correct class settings ===
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8      # Cityscapes 'thing' classes
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19  # Total classes (stuff + thing)

# === Solver / schedule ===
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = []

# === Output ===
cfg.OUTPUT_DIR = "./output_cityscapes_panoptic"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# === 3. Train ===
trainer = DefaultTrainer(cfg)
DetectionCheckpointer(trainer.model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
trainer.train()
