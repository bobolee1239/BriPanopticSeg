import os
import random
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes_panoptic import register_all_cityscapes_panoptic
from detectron2.utils.visualizer import Visualizer

# Path to your Cityscapes dataset root
CITYSCAPES_ROOT = "./Dataset"

# Clear any previously registered datasets (optional, for clean re-run)
for name in list(DatasetCatalog.keys()):
    DatasetCatalog.remove(name)
for name in list(MetadataCatalog.keys()):
    MetadataCatalog.remove(name)

# Register the Cityscapes panoptic dataset
register_all_cityscapes_panoptic(CITYSCAPES_ROOT)

# Get the val split
dataset_name = "cityscapes_fine_panoptic_val"
dataset_dicts = DatasetCatalog.get(dataset_name)
metadata = MetadataCatalog.get(dataset_name)

# Add 'isthing' field to segments_info
thing_ids = set(metadata.thing_dataset_id_to_contiguous_id.keys())
for d in dataset_dicts:
    for s in d["segments_info"]:
        s["isthing"] = s["category_id"] in thing_ids

# Visualize a few random samples
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
    out = visualizer.draw_dataset_dict(d)
    out_img = out.get_image()[:, :, ::-1]

    window_name = os.path.basename(d["file_name"])
    cv2.imshow(window_name, out_img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

cv2.destroyAllWindows()
