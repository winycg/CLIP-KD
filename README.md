# CLIP-KD
This repository contains the source code of CLIP-KD [CLIP-KD: An Empirical Study of CLIP Model Distillation].

## Install
```
pip install -r requirements-training.txt
pip install -r requirements-test.txt
```
## Dataset preparation

### Conceptual Captions 3M 

OpenCLIP reads a CSV file with two columns: a path to an image, and a text caption. The names of the columns are passed as an argument to `main.py`.

The script `src/data/gather_cc.py` will collect the Conceptual Captions 3M images. First, download the [Conceptual Captions 3M URLs](https://ai.google.com/research/ConceptualCaptions/download) and then run the script from our repository:
For easy notation, we rename `Train_GCC-training` as `cc3m_train`, and `Validation_GCC-1.1.0-Validation` as `cc3m_val`.
```bash
python src/data/gather_cc.py [path/to/cc3m/images/] [path/to/cc3m_train.tsv] [path/to/cc3m_val.tsv]
```

Our downloaded CC3M training set contains 2.89M images, and our CC3M validation set contains 13K images.


The generated `cc3m_train.csv` is:
```
title   filepath
XXXXXX  train/X/X.jpg
...     ...
```

The generated `cc3m_val.csv` is:
```
title   filepath
XXXXXX  val/X/X.jpg
...     ...
```

### Conceptual 12M 
The script `src/data/gather_cc12m.py` will collect the Conceptual 12M images. First, download the [Conceptual 12M URLs](https://storage.googleapis.com/conceptual_12m/cc12m.tsv) and then run the script from our repository:

```bash
python src/data/gather_cc12m.py [path/to/cc12m/images/] [path/to/cc12m.tsv]
```
The generated `cc12m.csv` is:
```
title   filepath
XXXXXX  train/X/X.jpg
...     ...
```

Our downloaded CC12M training set contains 9.97M images.



## Distill CLIP models

### Distillation with different strategies
The teacher is pretrained on CC3M+12M. Students are distilled on CC3M+12M.
| Role | Network |Method | ImageNet Acc| Train script |
| :----: | :----: | :----:  |:----:  |:----: |
|  Teacher | ViT-B/16|-| 36.99 |[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_B_16_baseline.sh)|
|  Student | ViT-T/16|Baseline|30.55|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_T_16_baseline.sh)|
|  Student | ViT-T/16| +CRD |31.94|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/CRD.sh)|
|  Student | ViT-T/16| +FD | 34.23|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/FD.sh)|
|  Student | ViT-T/16| +MFD | 34.09|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/MFD.sh)|
|  Student | ViT-T/16| +GD |31.54|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/GD.sh)|
|  Student | ViT-T/16| +ICL |33.11|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/ICL.sh)|
|  Student | ViT-T/16| +AFD |31.42|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/AFD.sh)|



### Supervised by ViT-B/16 as the teacher
The teacher is pretrained on CC3M+12M. Students are distilled on CC3M+12M.
| Role | Network |Method | ImageNet Acc| Train script | Download |
| :----:  | :----:  | :----:  |:----: |:----: | :----: |
|  Teacher | ViT-B/16|-| 36.99 |[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_B_16_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_cc3m_12m_ep32.log) |
|  Student | ViT-T/16|Baseline|30.55|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_T_16_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.log) |
|  Student | ViT-T/16| CLIP-KD |34.90|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/ViT_T_16_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_ViT-T-16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_ViT-T-16_cc3m_12m_ep32.log) |
|  Student | MobileViT-S |Baseline|32.60|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/mobilevit_s_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-mobilevit_s_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-mobilevit_s_cc3m_12m_ep32.log) |
|  Student | MobileViT-S |CLIP-KD|35.96|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/mobilevit_s_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-mobilevit_s_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-mobilevit_s_cc3m_12m_ep32.log) |
|  Student | Swin-T |Baseline|36.38|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/swin_tiny_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.log) |
|  Student | Swin-T |CLIP-KD|40.18|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/swin_tiny_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.log) |
|  Student | MobileNetV3 |Baseline|25.11|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/mobilenetv3_small_100_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm_mobilenetv3_small_100_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm_mobilenetv3_small_100_cc3m_12m_ep32.log) |
|  Student | MobileNetV3 |CLIP-KD|26.95|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/mobilenetv3_small_100_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm_mobilenetv3_small_100_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm_mobilenetv3_small_100_cc3m_12m_ep32.log) |
|  Student |  EfficientNet-B0 |Baseline|32.55|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/efficientnet_b0_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-efficientnet-b0_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-efficientnet-b0_cc3m_12m_ep32.log) |
|  Student |  EfficientNet-B0 |CLIP-KD|35.44|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/efficientnet_b0_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-efficientnet-b0_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-efficientnet-b0_cc3m_12m_ep32.log) |
|  Student |  ResNet-18 | Baseline|28.55 |[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/RN18_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-resnet18_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-resnet18_cc3m_12m_ep32.log) |
|  Student |  ResNet-18 | CLIP-KD|31.36|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/RN18_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-resnet18_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-resnet18_cc3m_12m_ep32.log) |

### Supervised by ResNet-101 as the teacher
The teacher is pretrained on CC3M+12M. Students are distilled on CC3M+12M.
| Role | Network |Method | ImageNet Acc| Train script | Download |
|:----:  | :----: | :----:  |:----:  |:----:  |:----:  |
|  Teacher |  ResNet-101 |-| 36.76 |[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/RN101_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_ep32.log) |
|  Student | MobileViT-S |Baseline|32.60|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/mobilevit_s_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-mobilevit_s_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-mobilevit_s_cc3m_12m_ep32.log) |
|  Student | MobileViT-S |CLIP-KD|34.97|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/RN101_KD/mobilevit_s_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-mobilevit_s_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-mobilevit_s_cc3m_12m_ep32.log) |
|  Student | Swin-T |Baseline|36.38|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/swin_tiny_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.log) |
|  Student | Swin-T |CLIP-KD|39.51|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/RN101_KD/swin_tiny_kd.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.log) |
|  Student | MobileNetV3 |Baseline|25.11|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/mobilenetv3_small_100_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm_mobilenetv3_small_100_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm_mobilenetv3_small_100_cc3m_12m_ep32.log) |
|  Student | MobileNetV3 |CLIP-KD|26.15|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/RN101_KD/mobilenetv3_small_100_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm_mobilenetv3_small_100_cc3m_12.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm_mobilenetv3_small_100_cc3m_12.log) |
|  Student |  EfficientNet-B0 |Baseline|32.55|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/efficientnet_b0_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-efficientnet-b0_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-efficientnet-b0_cc3m_12m_ep32.log) |
|  Student |  EfficientNet-B0 |CLIP-KD| 34.64|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/RN101_KD/efficientnet_b0_kd.sh)| [model]() \| [log]() |
|  Student |  ResNet-18 | Baseline|28.55 |[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/RN18_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-resnet18_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-resnet18_cc3m_12m_ep32.log) |
|  Student |  ResNet-18 | CLIP-KD|30.88|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/RN101_KD/RN18_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-efficientnet-b0_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-efficientnet-b0_cc3m_12m_ep32.log) |


### Transferred from Laion-400M
The teacher is pretrained on Laion-400M. Students are distilled on CC3M+12M.

| Role | Network | Method | ImageNet | Train script | Download |
| :----: | :----: | :----: | :----: | :----: | :----:|
|  Teacher |  ViT-L/14 |-| 72.8 |-|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_L_14-laion400m_ep32.pt)|
|  Student | ViT-B/16 |Baseline|37.0| [sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_B_16_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_cc3m_12m_ep32.log)|
|  Student | ViT-B/16 |CLIP-KD|57.5|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_L_14_KD_Laion/ViT_B_16_kd.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-L-14_laion400m_kd_ViT-B-16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-L-14_laion400m_kd_ViT-B-16_cc3m_12m_ep32.log)|
|  Student | ViT-T/16 |Baseline|30.6|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_T_16_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.log)|
|  Student | ViT-T/16 |CLIP-KD|40.9|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_L_14_KD_Laion/ViT_T_16_kd.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-L-14_laion400m_kd_ViT-T-16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-L-14_laion400m_kd_ViT-T-16_cc3m_12m_ep32.log)|


| Role | Network | Method | ImageNet | Train script | Download |
| :----: | :----: | :----: | :----: |:----:|:----:|
|  Teacher |  ViT-B/16 |-| 67.1 |-|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16-laion400m_e32.pt)|
|  Student | ViT-T/16 |Baseline|30.6| [sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_T_16_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.log)|
|  Student | ViT-T/16 |CLIP-KD|42.6|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD_Laion/ViT_T_16_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_laion400m_kd_ViT_T_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_laion400m_kd_ViT_T_16_cc3m_12m_ep32.log) |
|  Student | ResNet-50 |Baseline|35.3|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/RN50_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN50_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN50_cc3m_12m_ep32.log) |
|  Student | ResNet-50 |CLIP-KD|55.4| [sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD_Laion/RN50_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_laion400m_kd_RN50_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_laion400m_kd_RN50_cc3m_12m_ep32.log) |

### Evaluate pretrained models on more downstream tasks

Evaluation a pretrained model on MSCOCO and Flickr cross-retrieval and ImageNet variants (ImageNet-V2, ImageNet-Rendition and ImageNet-Sketch) classification. Please refer to [eval_coco.sh](https://github.com/winycg/CLIP-KD/blob/main/script/eval/eval_coco.sh) and [eval_flickr.sh](https://github.com/winycg/CLIP-KD/blob/main/script/eval/eval_flickr.sh).


## Acknowledgement
Our codebase is bulit over [open_clip](https://github.com/mlfoundations/open_clip), an open-source codebase to run CLIP models.

We would appreciate it if our paper and repo are helpful to you!
```
@inproceedings{yang2024clip,
  title={CLIP-KD: An Empirical Study of CLIP Model Distillation},
  author={Yang, Chuanguang and An, Zhulin and Huang, Libo and Bi, Junyu and Yu, Xinqiang and Yang, Han and Diao, Boyu and Xu, Yongjun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```