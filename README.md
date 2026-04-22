# A Novel Fine-Tuned CLIP-OOD Detection Method with Double Loss Constraint Through Optimal Transport Semantic Alignment
AAAI-26 Accepted Paper：

**[A Novel Fine-Tuned CLIP-OOD Detection Method with Double Loss Constraint Through Optimal Transport Semantic Alignment](https://ojs.aaai.org/index.php/AAAI/article/view/38572)**

Heng-yang Lu, Xin Guo, Shuai Feng, Wenyu Jiang, Yuntao Du, Chang Xia, Chenyou Fan

Abstract: *Detecting Out-Of-Distribution (OOD) samples in image classification is crucial for model reliability. With the rise of Vision-Language Models (VLMs), CLIP-OOD has become a research hotspot. However, we observe the Low Focus Attention phenomenon from the image encoders of CLIP, which means the attention of image encoders often spreads to non-in-distribution regions. This phenomenon comes from the semantic misalignment and inter-class feature confusion. To address these issues, we propose a novel fine-tuned OOD detection method with the Double loss constraint based on Optimal Transport (DOT-OOD). DOT-OOD integrates the Double Loss Constraint (DLC) module and Optimal Transport (OT) module. The DLC module comprises the Aligned Image-Text Concept Matching Loss and the Negative Sample Repulsion Loss, which respectively (1) focus on the core semantics of ID images and achieve cross-modal semantic alignment, (2) expand inter-class distances and enhance discriminative. While the OT module is introduced to obtain enhanced image feature representations. Extensive experimental results show that in the 16-shot scenario of the ImageNet-1k benchmark, DOT-OOD reduces the FPR95 by over 10% and improves the AUROC from 94.48% to 96.57% compared with SOTAs.*

![The Attention Distribution](introduction.png)

## DataSets

Dataset source can be downloaded here.

- [ImageNet](https://www.image-net.org/). The ILSVRC 2012 dataset as In-distribution (ID) dataset. The training subset we used is [this file](datalists/imagenet2012_train.txt).
- [Texture](https://www.robots.ox.ac.uk/~vgg/data/dtd/). We rule out four classes that coincides with ImageNet. The filelist used in the paper is [here](datalists/texture.txt).
- [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf). Follow the instructions in the [link](https://github.com/deeplearning-wisc/large_scale_ood) to prepare the iNaturalist OOD dataset.
- [SUN](https://faculty.cc.gatech.edu/~hays/papers/sun.pdf). . Follow the instructions in the [line](https://github.com/YongHyun-Ahn/LINe-Out-of-Distribution-Detection-by-Leveraging-Important-Neurons)
- [Places](http://olivalab.mit.edu/Papers/Places-PAMI2018.pdf). Follow the instructions in the [line](https://github.com/YongHyun-Ahn/LINe-Out-of-Distribution-Detection-by-Leveraging-Important-Neurons)

## Prepare

Our code run on environment, the required list in [requirements.txt](requirements.txt). Please install the list before running.

## Fine-tunning

The Fine-tunning code is in [train.py](train.py). Run the code for fine-tunning:
```bash
python train.py
```

## OOD Detection

We provide four OOD Detection method: [CSP](ood_detection_csp.py)、[NegLabel](ood_detection_neglabel.py)、 [MCM](ood_detection_mcm.py)、[GL-MCM](ood_detection_glmcm.py). For OOD Detection, you can run:
```bash
python ood_detection_csp.py
python ood_detection_neglabel.py
python ood_detection_glmcm.py
python ood_detection_mcm.py
```

## Citation

```bash
@inproceedings{lu2026novel,
  title={A Novel Fine-Tuned CLIP-OOD Detection Method with Double Loss Constraint Through Optimal Transport Semantic Alignment},
  author={Lu, Heng-yang and Guo, Xin and Feng, Shuai and Jiang, Wenyu and Du, Yuntao and Xia, Chang and Fan, Chenyou},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={18},
  pages={15448--15456},
  year={2026},
  doi={10.1609/aaai.v40i18.38572}
}
```