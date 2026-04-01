
<a id="top"></a>
<div align="center">
  <img src="./assets/logo.png" width="500"> 
  <h1>(ACM MM 2025) OFFSET: Segmentation-based Focus Shift Revision for Composed Image Retrieval</h1>

  <p>
    <a href="https://acmmm2025.org/"><img src="https://img.shields.io/badge/ACM_MM-2025-blue.svg?style=flat-square" alt="ACM MM 2025"></a>
    <a href="https://arxiv.org/abs/2507.05631"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2507.05631-b31b1b.svg"></a>
    <a href="https://dl.acm.org/doi/10.1145/3746027.3755366"><img alt='Paper' src="https://img.shields.io/badge/Paper-dl.acm-green.svg?style=flat-square"></a>
    <a href="https://zivchen-ty.github.io/OFFSET.github.io/"><img alt='page' src="https://img.shields.io/badge/Website-orange?style=flat-square"></a>
        <a href="https://zivchen-ty.github.io"><img src="https://img.shields.io/badge/Author Page-blue.svg" alt="Author Page"></a>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
    <img src="https://img.shields.io/badge/python-3.8.10-blue?style=flat-square" alt="Python">
    <a href="https://github.com/"><img alt='stars' src="https://img.shields.io/github/stars/ZivChen-Ty/offset?style=social"></a>
  </p>


  <p>
    <b>Accepted by ACM MM 2025:</b> A novel network designed to address visual inhomogeneity and text-priority biases in Composed Image Retrieval (CIR) through dominant portion segmentation and textually guided focus revision.
  </p>
</div>

## 📌 Introduction
Welcome to the official repository for **OFFSET** (Segmentation-based Focus Shift Revision for Composed Image Retrieval). 

Existing CIR approaches often overlook the inhomogeneity between dominant and noisy portions in visual data, leading to query feature degradation. Furthermore, they ignore the priority of textual data in the image modification process, resulting in a visual focus bias. **OFFSET** tackles these limitations using a focus mapping-based feature extractor and a textually guided focus revision module, achieving State-of-the-Art (SOTA) performance across multiple datasets.

[⬆ Back to top](#top)

## 📢 News
- **[2026-03-20]** 🚀 We migrate the all training and evaluation codes of OFFSET from Google Drive to a GitHub repository. 
* **[2025-07-05]** 🔥 OFFSET has been accepted by **ACM MM 2025**.
* **[2024-12-26]** 📍 We release the main codes and data of OFFSET!


[⬆ Back to top](#top)

## ✨ Key Features
Our framework introduces key innovative modules to achieve precise multimodal semantic alignment:

* 🔍 **Dominant Portion Segmentation**: Utilizes visual language models to generate image captions as a role-supervised signal, dividing dominant and noisy regions to effectively mask noise information.
* 🔗 **Dual Focus Mapping**: Features Visual Focus Mapping (VFM) and Textual Focus Mapping (TFM) branches. Guided by the dominant segmentation, it accomplishes adaptive focus mapping on both visual and textual data.
* 🧩 **Textually Guided Focus Revision**: Utilizes the modification requirements embedded in the textual feature to perform adaptive focus revision on the reference image, enhancing the perception of the modification focus.
* 🏆 **SOTA Performance**: Demonstrates superior generalization and achieves remarkable improvements across both fashion-domain (FashionIQ, Shoes) and open-domain (CIRR) datasets.

[⬆ Back to top](#top)

## 🏗️ Architecture

<p align="center">
  <img src="assets/OFFSET-MM25.png" alt="OFFSET architecture" width="900">
  <figcaption><strong>Figure 1.</strong> The overall architecture of OFFSET. It consists of three key modules: Dominant Portion Segmentation, Dual Focus Mapping, and Textually Guided Focus Revision.</figcaption>
</p>

[⬆ Back to top](#top)

## 📊 Experiment Results

OFFSET consistently outperforms existing baselines on widely-used datasets, surpassing strong competitors like DQU-CIR and ENCODER.

### 1. FashionIQ & Shoes Datasets
*(Evaluated using Recall@K)*
<div align="center">
  <img src="assets/results-fiq.png" alt="FashionIQ and Shoes Results" height="330" style="object-fit:contain;">
  <img src="assets/results-shoes.png" alt="FashionIQ and Shoes Results" height="330" style="object-fit:contain;">
</div>

### 2. CIRR Dataset
*(Evaluated using R@K and R_subset@K)* <p align="center">
  <img src="assets/results-cirr.png" alt="CIRR Results" width="700">
</p>

[⬆ Back to top](#top)

---

## 📑 Table of Contents

- [📌 Introduction](#-introduction)
- [📢 News](#-news)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [📊 Experiment Results](#-experiment-results)
- [📂 Repository Structure](#-repository-structure)
- [🚀 Installation](#-installation)
- [📂 Data Preparation](#-data-preparation)
  - [Shoes](#shoes)
  - [FashionIQ](#fashioniq)
  - [CIRR](#cirr)
- [🏃‍♂️ Quick Start](#️-quick-start)
  - [1. Training the Model](#1-training-the-model)
  - [2. Evaluating the Model](#2-evaluating-the-model)
  - [3. Test for CIRR](#3-test-for-cirr)
- [📝 Citation](#-citation)
- [🤝 Acknowledgements](#-acknowledgements)
- [✉️ Contact](#️-contact)

---

## 📂 Repository Structure
Our codebase is highly modular. Here is a brief overview of the core files:

```text
OFFSET/
├── cirr_test_submission.py# 📄 CIRR submission file generator
├── datasets.py            # 📚 Dataset loader and preprocessing
├── model_OFFSET.py        # 🧠 OFFSET model architecture and forward pass
├── test.py                # 🧪 Evaluation/Test entry point
├── train.py               # 🚀 Training entry point
├── utils.py               # 🛠️ Utility functions (metrics, helper methods)
└── README.md              # 📝 Documentation and result visualization
```

This section helps users quickly locate the core components and get started with development.

## 🚀 Installation

**1. Clone the repository**
```bash
git clone https://github.com/ZivChen-Ty/OFFSET.git
cd OFFSET
```

**2. Setup Environment**
We recommend using Conda to manage your environment:

```bash
conda create -n offset_env python=3.8.10
conda activate offset_env

# Install PyTorch (Ensure it matches your CUDA version. Tested on PyTorch 2.0.0, NVIDIA A40 48G)
pip install torch==2.0.0 torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install required packages
pip install -r requirements.txt
```


## 📂 Data Preparation

#### 🛟【OURS】Pre-computed Dominant Portion Segmentation Data (Official Release)
*The dominant portion segmentation data of OFFSET is available at [Google Drive](https://drive.google.com/file/d/1_tdFuGec__NTv-_DjUZ66dCwVgBEHxcX/view?usp=sharing).*  
> 🔥 This is our official released data for result reproduction.

OFFSET is evaluated on FashionIQ, Shoes, and CIRR. Please download the datasets from their official sources and arrange them as follows.

#### Shoes

Download the Shoes dataset following the instructions in
the [official repository](https://github.com/XiaoxiaoGuo/fashion-retrieval/tree/master/dataset).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── Shoes
│   ├── captions_shoes.json
│   ├── eval_im_names.txt
│   ├── relative_captions_shoes.json
│   ├── train_im_names.txt
│   ├── [womens_athletic_shoes | womens_boots | ...]
|   |   ├── [0 | 1]
|   |   ├── [img_womens_athletic_shoes_375.jpg | descr_womens_athletic_shoes_734.txt | ...]
```

#### FashionIQ

Download the FashionIQ dataset following the instructions in
the [official repository](https://github.com/XiaoxiaoGuo/fashion-iq).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── FashionIQ
│   ├── captions
|   |   ├── cap.dress.[train | val | test].json
|   |   ├── cap.toptee.[train | val | test].json
|   |   ├── cap.shirt.[train | val | test].json

│   ├── image_splits
|   |   ├── split.dress.[train | val | test].json
|   |   ├── split.toptee.[train | val | test].json
|   |   ├── split.shirt.[train | val | test].json

│   ├── dress
|   |   ├── [B000ALGQSY.jpg | B000AY2892.jpg | B000AYI3L4.jpg |...]

│   ├── shirt
|   |   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]

│   ├── toptee
|   |   ├── [B0000DZQD6.jpg | B000A33FTU.jpg | B000AS2OVA.jpg | ...]
```

#### CIRR

Download the CIRR dataset following the instructions in the [official repository](https://github.com/Cuberick-Orion/CIRR).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRR
│   ├── train
|   |   ├── [0 | 1 | 2 | ...]
|   |   |   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]

│   ├── dev
|   |   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]

│   ├── test1
|   |   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]

│   ├── cirr
|   |   ├── captions
|   |   |   ├── cap.rc2.[train | val | test1].json
|   |   ├── image_splits
|   |   |   ├── split.rc2.[train | val | test1].json
```

## 🏃‍♂️ Quick Start

### 1\. Training the Model

Train OFFSET on Shoes, FashionIQ, or CIRR using the `train.py` script.

```bash
python3 train.py \
    --model_dir ./checkpoints/ \
    --dataset {shoes, fashioniq, cirr} \
    --cirr_path "path/to/CIRR" \
    --fashioniq_path "path/to/FashionIQ" \
    --shoes_path "path/to/Shoes"
```

### 2\. Test for CIRR

To generate the predictions file for uploading to the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/) using our model, please execute the following command:

```bash
python src/cirr_test_submission.py model_path
```

*(Where `model_path` is the path to the OFFSET model checkpoint on CIRR)*



## 🤝 Acknowledgements

This project builds upon recent advancements in Composed Image Retrieval and Vision-Language pre-training. We express our sincere gratitude to the open-source community for their contributions. Supported in part by the National Natural Science Foundation of China.

## ✉️ Contact

If you have any questions, feel free to [open an issue](https://www.google.com/search?q=https://github.com/ZivChen-Ty/OFFSET/issues) or reach out to me zivczw@gmail.com ☺️

## 🔗 Related Projects

*Ecosystem & Other Works from our Team*

<table style="width:100%; border:none; text-align:center; background-color:transparent;">
    <tr style="border:none;">
   <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/airknow-logo.png" alt="HABIT" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>Air-Know (CVPR'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://zhihfu.github.io/Air-Know.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/zhihfu/Air-Know" target="_blank">Code</a> | 
        <!-- <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37608" target="_blank">Paper</a> -->
      </span>
    </td>
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/habit-logo.png" alt="HABIT" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>HABIT (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://lee-zixu.github.io/HABIT.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/Lee-zixu/HABIT" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37608" target="_blank">Paper</a>
      </span>
    </td>
      <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/habit-logo.png" alt="HABIT" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>HABIT (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://lee-zixu.github.io/HABIT.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/Lee-zixu/HABIT" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37608" target="_blank">Paper</a>
      </span>
    </td>
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/retrack-logo.png" alt="ReTrack" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>ReTrack (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://lee-zixu.github.io/ReTrack.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/Lee-zixu/ReTrack" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/39507" target="_blank">Paper</a>
      </span>
    </td>
    </tr>
  <tr style="border:none;">
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/intent-logo.png" alt="INTENT" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>INTENT (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://zivchen-ty.github.io/INTENT.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/ZivChen-Ty/INTENT" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/39181" target="_blank">Paper</a>
      </span>
    </td>  
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/hud-logo.png" alt="HUD" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>HUD (ACM MM'25)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://zivchen-ty.github.io/HUD.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/ZivChen-Ty/HUD" target="_blank">Code</a> | 
        <a href="https://dl.acm.org/doi/10.1145/3746027.3755445" target="_blank">Paper</a>
      </span>
    </td>
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/encoder-logo.png" alt="ENCODER" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>ENCODER (AAAI'25)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://sdu-l.github.io/ENCODER.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/Lee-zixu/ENCODER" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/32541" target="_blank">Paper</a>
      </span>
    </td>
  </tr>
</table>


## 📝⭐️ Citation

If you find our work or this code useful in your research, please consider leaving a **Star**⭐️ or **Citing**📝 our paper 🥰. Your support is our greatest motivation\!

```bibtex
@inproceedings{OFFSET, 
  title = {OFFSET: Segmentation-based Focus Shift Revision for Composed Image Retrieval}, 
  author = {Chen, Zhiwei and Hu, Yupeng and Li, Zixu and Fu, Zhiheng and Song, Xuemeng and Nie, Liqiang}, 
  booktitle = {Proceedings of the ACM International Conference on Multimedia}, 
  pages = {6113–6122}, 
  year = {2025}
}
```


## 🫡 Support & Contributing

We welcome all forms of contributions\! If you have any questions, ideas, or find a bug, please feel free to:

  - Open an [Issue](https://github.com/ZivChen-Ty/OFFSET/issues) for discussions or bug reports.
  - Submit a [Pull Request](https://github.com/ZivChen-Ty/OFFSET/pulls) to improve the codebase.

[⬆ Back to top](#top)

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="500" alt="OFFSET Demo">

  <br><br>

  <a href="https://github.com/ZivChen-Ty/OFFSET">
    <img src="https://img.shields.io/badge/⭐_Star_US-000000?style=for-the-badge&logo=github&logoColor=00D9FF" alt="Star">
  </a>
  <a href="https://github.com/ZivChen-Ty/OFFSET/issues">
    <img src="https://img.shields.io/badge/🐛_Report_Issues-000000?style=for-the-badge&logo=github&logoColor=FF6B6B" alt="Issues">
  </a>
  <a href="https://github.com/ZivChen-Ty/OFFSET/pulls">
    <img src="https://img.shields.io/badge/🧐_Pull_Requests-000000?style=for-the-badge&logo=github&logoColor=4ECDC4" alt="Pull Request">
  </a>

  <br><br>
<a href="https://github.com/ZivChen-Ty/OFFSET">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00D9FF&center=true&vCenter=true&width=500&lines=Thank+you+for+visiting+OFFSET!;Looking+forward+to+your+attention!" alt="Typing SVG">
  </a>
</div>
