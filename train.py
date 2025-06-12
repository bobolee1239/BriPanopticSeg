from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from register_cityscapes_panoptic import register_cityscapes_panoptic

# Register dataset
register_cityscapes_panoptic()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))

cfg.DATASETS.TRAIN = ("cityscapes_fine_panoptic_train",)
cfg.DATASETS.TEST = ("cityscapes_fine_panoptic_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = ""  # ← training from scratch!
cfg.SOLVER.IMS_PER_BATCH = 2  # you can increase this if you have enough VRAM
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 9000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19  # Cityscapes has 19 classes
cfg.OUTPUT_DIR = "./output_cityscapes"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

