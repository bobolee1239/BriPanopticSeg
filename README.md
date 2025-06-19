# Panoptic FPN for Cityscapes

## 📜 Introduction

I initially planned to explore panoptic segmentation using Detectron2 and the Cityscapes dataset. However, I realized that Detectron2’s implementation of PanopticFPN wasn’t fully compatible with the structure and needs of the Cityscapes dataset. So, I decided to collaborate with GPT and build a custom training pipeline for PanopticFPN tailored specifically for Cityscapes.

This project implements a **Panoptic Segmentation** pipeline based on the paper [*Panoptic Feature Pyramid Networks* (Kirillov et al., 2019)](https://arxiv.org/pdf/1901.02446). It combines both **instance segmentation** and **semantic segmentation** into a unified framework using PyTorch Lightning.

## 🔧 Features

* ✅ PanopticFPN architecture built on top of ResNet50 + FPN backbone
* ✅ Separate semantic and instance segmentation heads
* ✅ Fully integrated training routine with loss logging and visualization
* ✅ Visualization on TensorBoard for:

  * Instance segmentation results
  * Semantic segmentation masks
  * Fused panoptic predictions (instance + semantic)
* ✅ NMS applied to instance predictions during inference
* ✅ Fine-grained control of visualization frequency


## 📂 Dataset Preparation: Cityscapes + Panoptic Annotations

This project expects the Cityscapes dataset in the standard folder structure with panoptic annotations. Here's how you can prepare the dataset:

1. Download Cityscapes

Register and download the following datasets from Cityscapes Official Site:

`leftImg8bit_trainvaltest.zip`

`gtFine_trainvaltest.zip`

`gtCoarse.zip` (optional, if you want to use coarse annotations)

`gtFine_panoptic_trainId2labelId.zip` (may require generation if not downloaded)

Unzip everything so you have a structure like:
```bash
Cityscapes/
├── gtFine/
│   ├── train/
│   ├── val/
│   └── test/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
```

2. Install cityscapesscripts

```bash
$ pip install cityscapesscripts
```
Alternatively, you can clone it manually:
```bash
$ git clone https://github.com/mcordts/cityscapesScripts.git
$ cd cityscapesScripts
$ pip install -e .
```
3. Generate Panoptic Ground Truths

`CITYSCAPES_DIR=/path/to/Cityscapes`

```bash
$ python cityscapesscripts/preparation/createPanopticImgs.py \
  --cityscapesPath "$CITYSCAPES_DIR" \
  --outputDir "$CITYSCAPES_DIR/gtFine"
```

## 📂 Project Structure

```bash
BriPanopticSeg/
├── Model/
│   └── PanopticFpn.py            # Network architecture
├── Train/
│   └── PanopticTrainRoutine.py   # Training & visualization logic
├── ...
```

## 🖼️ Visualization

Visual logs during validation include:

* **Instance predictions**: things with bounding boxes + masks
* **Semantic segmentation**: all categories with colored overlays
* **Fused panoptic results**: semantic mask base with overlaid instance predictions (NMS-filtered)

These are automatically logged to TensorBoard every `n` steps.

## 🧠 Dependencies

* PyTorch ≥ 2.0
* torchvision
* pytorch-lightning
* numpy
* matplotlib

Install with:

```bash
$ pip install -r requirements.txt
```

## 📘️ Reference

* [Panoptic Feature Pyramid Networks](https://arxiv.org/pdf/1901.02446)

