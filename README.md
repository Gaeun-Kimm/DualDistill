# DualDistill: A Unified Cross-Modal Knowledge Distillation Framework for Camera-Based BEV Representation

</div>

[https://github.com/user-attachments/assets/6853248f-23f9-46b2-b4b2-559a67588ce7](https://github.com/user-attachments/assets/6853248f-23f9-46b2-b4b2-559a67588ce7)

> DualDistill: A Unified Cross-Modal Knowledge Distillation Framework for Camera-Based BEV Representation, BMVC 2025
> 
> - [Paper in OpenReview](https://openreview.net/forum?id=8sek44Vz1p#discussion)

# News

- [2025/07]: DualDistill is accepted at BMVC 2025 🔥
</br>

# Abstract

Cross-modal knowledge distillation has drawn much attention to camera-based bird’s-eye-view (BEV) models, aiming to narrow the performance gap with their LiDAR-based counterparts. However, distilliing knowledge from a LiDAR-based teacher is not easy due to the discrepancy between sensor modalities. In this work, we introduce DualDistill, a unified cross-modal knowledge distillation framework to address this challenge. We propose an attention-guided orthogonal alignment (AOA) to align student features with the teacher's representations while preserving useful information. This alignment is integrated into a multi-scale feature distillation with adaptive region weighting scheme. In addition, we introduce a cross-head response distillation (CRD) to enforce consistency in BEV representations by comparing the predictions of the teacher and the aligned student. We evaluate our method on the nuScenes dataset. Comprehensive experiments show that our method significantly improves camera-based BEV models and outperforms recent cross-modal knowledge distillation techniques.

# Methods

![image.png](https://github.com/user-attachments/assets/d6a3561e-3f0d-403d-b150-067fa43bc5ce)

# Getting Started

- [Installation](https://www.notion.so/docs/install.md)
- [Prepare Dataset](https://www.notion.so/docs/prepare_dataset.md)
- [Run and Eval](https://www.notion.so/docs/getting_started.md)

# Model Zoo


# Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{,
  title={},
  author={}
  journal={},
  year={}
}
```