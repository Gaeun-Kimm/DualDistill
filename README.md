# DualDistill: A Unified Cross-Modal Knowledge Distillation Framework for Camera-Based BEV Representation

<div align="center">

</div>

[https://github.com/user-attachments/assets/7ffcfbec-1a66-4b52-80c4-c127b685e535](https://github.com/user-attachments/assets/7ffcfbec-1a66-4b52-80c4-c127b685e535)

> DualDistill: A Unified Cross-Modal Knowledge Distillation Framework for Camera-Based BEV Representation, BMVC 2025
> 
> - [Paper in OpenReview](https://openreview.net/forum?id=8sek44Vz1p#discussion)

# News

---

- [2025/07]: DualDistill is accepted at BMVC 2025 🔥
</br>

# Abstract

---

Cross-modal knowledge distillation has drawn much attention to camera-based bird’s-eye-view (BEV) models, aiming to narrow the performance gap with their LiDAR-based counterparts. However, distilliing knowledge from a LiDAR-based teacher is not easy due to the discrepancy between sensor modalities. In this work, we introduce DualDistill, a unified cross-modal knowledge distillation framework to address this challenge. We propose an attention-guided orthogonal alignment (AOA) to align student features with the teacher's representations while preserving useful information. This alignment is integrated into a multi-scale feature distillation with adaptive region weighting scheme. In addition, we introduce a cross-head response distillation (CRD) to enforce consistency in BEV representations by comparing the predictions of the teacher and the aligned student. We evaluate our method on the nuScenes dataset. Comprehensive experiments show that our method significantly improves camera-based BEV models and outperforms recent cross-modal knowledge distillation techniques.

# Methods

---

![image.png](https://github.com/user-attachments/assets/8292f5bc-9cc5-423e-9ee0-e02e352538ec/image.png)

# Getting Started

---

### Installation

You can build the docker image by

```
docker build - < Dockerfile
```

Once the docker is running, run

```
pip install -v -e .
```

### Data Preparation

You can download nuScenes 3D detection data from the official webset and unzip alll zip files.

It is recommended to symlink the dates root to $DualDistill/data.

```
DualDistill
├── configs
├── docs
├── mmdet3d
├── requirements
├── scripts
├── tools
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

First, run the following command to get the .pkl files.

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes –virtual --extra-tag nuscenes
```

```
DualDistill
├── configs
├── docs
├── mmdet3d
├── requirements
├── scripts
├── tools
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
|   |   ├── nuscenes_infos_train.pkl
|   |   ├── nuscenes_infos_val.pkl
|   |   ├── nuscenes_infos_test.pkl
```

Then, we run the following command to generate the adjacent information for BEVDepth.

```
python tools/data_converter/prepare_nuscenes_for_bevdet4d.py
```

```
DualDistill
├── configs
├── docs
├── mmdet3d
├── requirements
├── scripts
├── tools
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
|   |   ├── nuscenes_infos_train.pkl
|   |   ├── nuscenes_infos_val.pkl
|   |   ├── nuscenes_infos_test.pkl
|   |   ├── nuscenes_infos_train_4d_interval3_max60.pkl
|   |   ├── nuscenes_infos_val_4d_interval3_max60.pkl
|   |   ├── nuscenes_infos_test_4d_interval3_max60.pkl
```

### Training and Evaluation

Train a distilled BEVDepth with CenterPoint as teacher

```
./scripts/teacher_to_bevdepth4d/centerpoint2bevdepth.sh
```

Train a distilled BEVDepth with MVP as teacher

```
./scripts/teacher_to_bevdepth4d/mvp2bevdepth.sh
```

For single-GPU evaluation, you can run following command

```
python tools/test.py configs/lidar2camera_bev_distillation/teacher_to_bevformer/lidarformer_to_bevformer_nus_1x1conv_r50.py outputs/lidarformer_to_bevformer_r50/epoch_24.pth --eval mAP
```

# Model Zoo

| Student | Teacher | Backbone | mAP | NDS | Download |
| --- | --- | --- | --- | --- | --- |
| BEVDepth |  | ResNet-50 | 35.1 | 47.5 |  |
| BEVDepth | CenterPoint | ResNet-50 | 40.5 | 51.4 |  |
| BEVDepth | MVP | ResNet-50 | 41.5 | 51.9 |  |
| BEVDepth |  | ResNet-100 | 41.2 | 53.5 |  |
| BEVDepth | CenterPoint | ResNet-100 | 45.5 | 55.6 |  |
| BEVDepth | MVP | ResNet-100 | 46.6 | 56.3 |  |
