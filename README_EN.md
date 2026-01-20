# Enhancement of Super-Resolution Models for Computed Tomography Images Using Fine-Tuning  

## Table of Contents
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Process Stages](#process-stages)  
4. [Methodological Choices](#methodological-choices)  
5. [Evaluation Metrics](#evaluation-metrics)  
6. [Project Structure](#project-structure)  
7. [Installation and Execution](#installation-and-execution)  

---

## Introduction

This repository contains the code developed within the scope of an academic project focused on enhancing low-dose computed tomography (CT) images through super-resolution techniques based on deep learning. The central motivation is the possibility of improving the visual and structural quality of medical images without increasing patient exposure to ionizing radiation, relying exclusively on computational post-processing.

The project is based on adapting super-resolution models pre-trained on natural images to the specific domain of computed tomography, using the fine-tuning technique. The main objective is to evaluate the impact of this fine adjustment on model performance by comparing direct inference with inference after fine-tuning, and to analyze whether specialization to the medical domain contributes to perceptual quality improvement and greater structural fidelity in the reconstructed images. All implementation was carried out using free and open-source tools, aiming at reproducibility and accessibility.

---

## Dataset

The dataset used in this project is LoDoPaB-CT (Low-Dose Parallel Beam – Computed Tomography), commonly employed in studies related to reconstruction and enhancement of low-dose computed tomography images. This dataset is composed of simulated human chest CT images, enabling the formation of corresponding low-resolution and high-resolution image pairs.

Within the context of this project, the dataset is organized into training, validation, and test subsets, allowing both supervised fine-tuning and quantitative evaluation of results. The original images are provided in HDF5 format and undergo a processing stage in which they are converted to BMP format, making them compatible with the super-resolution models employed.

---

## Process Stages

The overall workflow of the project is organized into the following stages:

1. **Data preparation**  
   - Organization of training, validation, and test sets  
   - Conversion of images from HDF5 to BMP format  

2. **Direct inference**  
   - Application of pre-trained models without additional adjustment  
   - Generation of reference super-resolved images (baseline)  

3. **Model fine-tuning**  
   - Supervised adjustment of the final network weights  
   - Specialization of the models to the low-dose CT domain  

4. **Post fine-tuning inference**  
   - Generation of reconstructed images using the fine-tuned models  

5. **Quantitative evaluation**  
   - Computation of quality metrics  
   - Comparison between direct inference and post fine-tuning inference  

---

## Methodological Choices

The main decisions adopted in the development of this project were:

- Use of pre-trained models, reducing computational cost and training time  
- Application of shallow fine-tuning, adjusting only the final layers of the networks  
- Use of the LoDoPaB-CT dataset  
- Explicit separation between training, validation, and test sets to ensure consistent evaluation  

These choices aim to balance performance, reproducibility, and computational feasibility.

---

## Evaluation Metrics

The performance evaluation of the super-resolution models is carried out using the following metrics:

**PSNR (Peak Signal-to-Noise Ratio)**  
- Evaluates the signal-to-noise ratio between the reconstructed image and the reference image. Higher values indicate better reconstruction quality.

**SSIM (Structural Similarity Index Measure)**  
- Measures the structural similarity between the super-resolved image and the reference image, considering luminance, contrast, and structure.

**PI (Perceptual Index)**  
- A perceptual metric that combines visual quality information to assess the naturalness of reconstructed images. Lower values indicate better perceptual quality.

These metrics allow a complementary analysis of structural fidelity and visual quality of the reconstructed images.

---

## Project Structure

The organization of files and directories in the repository follows the structure below.  
**Some project folders contain a file named `instruction.md`**, which describes the purpose of that folder, the files it stores, and what happens to its contents throughout the execution of the pipeline.

```text
Fine-Tuning-for-Computed-Tomography/
├─ Datasets/
│  ├─ test/         - Dataset containing test images
│  ├─ train/        - Dataset containing training images
│  ├─ validation/   - Dataset containing validation images
├─ Models/
│  ├─ checkpoints/  - Models generated during training
├─ Others/
│  ├─ Logs/         - Execution log records
│  └─ Metrics/      - Computed metric results
├─ src/
│  ├─ util/
│  │  ├─ hat_arch.py            - Dependencies for HAT execution
│  │  ├─ util_basicsr.py        - Adjustments to the BasicSR library
│  │  └─ utils.py               - Auxiliary functions
│  ├─ fine_tune.py              - Model training (fine-tuning)
│  ├─ inference.py              - Model inference
│  ├─ main.py                   - Main orchestration script
│  ├─ metrics.py                - Metric computation
│  ├─ namelist.py               - Execution configuration
│  └─ pre_processing.py         - HDF5 to BMP conversion
├─ environment.yml              - Conda environment dependencies
└─ README.md
```

---
## Installation and Execution

The installation and execution instructions follow the steps below:

### 1. Install Anaconda
- https://www.anaconda.com/download/

### 2. Create the virtual environment
```python
conda create -y -n .venv_FT python=3.10
```

### 3. Activate the virtual environment
```python
conda activate .venv_FT
```

### 4. Install libraries from environment.yml
```python
conda env update -n .venv_FT -f environment.yml
```
### 5. Add the dataset to the Datasets folder
- Follow the instructions described in the `instruction.md` files inside the Datasets folder.

### 6. Edit the `namelist.py` file
- Located in the `src`folder, fill in the required information.

### 7. Run the code
```python
python src/main.py
```

### 8. Monitor execution through log records
- Available in the folder: `Others/logs`

---
