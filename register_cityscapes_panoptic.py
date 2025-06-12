from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes_panoptic import register_all_cityscapes_panoptic

CITYSCAPES_ROOT = "/home/brian-lee/ssd1/Dataset/Cityscapes"

# Remove previous registration if it exists
for split in ["train", "val"]:
    base = f"cityscapes_fine_panoptic_{split}"
    if base in DatasetCatalog:
        print(f"🧹 Removed default registration for: {base}")
        DatasetCatalog.remove(base)
        MetadataCatalog.remove(base)
    for suffix in ["_separated", "_stuffonly"]:
        full_name = base + suffix
        if full_name in DatasetCatalog:
            print(f"🧹 Removed default registration for: {full_name}")
            DatasetCatalog.remove(full_name)
            MetadataCatalog.remove(full_name)

# Register fresh
register_all_cityscapes_panoptic(CITYSCAPES_ROOT)
print("✅ Re-registered all Cityscapes panoptic datasets.")

# (Optional) Print to confirm
print("✅ Registered datasets:")
for name in DatasetCatalog.list():
    if "cityscapes" in name:
        print("  🔹", name)
