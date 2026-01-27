# Enhancement of Super-Resolution Models for Computed Tomography Images Using Fine-Tuning  

## Table of Contents
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Process Stages](#process-stages)  
4. [Methodological Choices](#methodological-choices)  
5. [Evaluation Metrics](#evaluation-metrics)
6. [Quantitative Results](#quantitative-results)
7. [Project Structure](#project-structure)  
8. [Installation and Execution](#installation-and-execution)  

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

## Quantitative Results

The quantitative results show that the fine-tuning process promotes consistent improvements compared to direct inference, both in the original-resolution domain and in the reduced-resolution domain. A significant increase in the mean PSNR and SSIM values is observed, accompanied by a reduction in the Perceptual Index (PI), indicating simultaneously higher structural fidelity and better perceptual quality of the reconstructed images. The associated boxplots reinforce this trend by showing lower result dispersion and a shift in the distributions in favor of the fine-tuned models when compared to models without fine-tuning and to the reference FBP method.

The results presented below report the mean and standard deviation of the PSNR, SSIM, and PI metrics for the different evaluated methods, considering both the original and reduced resolution domains. It can be observed that the models subjected to the fine-tuning process consistently outperform direct inference and the FBP reference method.

##### Table 1 – Results in the original resolution domain (362×362 pixels)

| Method             | Training Resoluiton | PSNR (↑)           | SSIM (↑)           | PI (↓)            |
|----------------------|--------------------------|--------------------|--------------------|-------------------|
| FBP                  | –                        | 18,81 ± 1,83       | 0,34 ± 0,09        | 4,11 ± 1,11       |
| Real-ESRGAN (pre)    | –                        | 19,24 ± 2,07       | 0,41 ± 0,09        | 4,68 ± 1,01       |
| HAT (pre)            | –                        | 17,12 ± 2,33       | 0,31 ± 0,09        | 3,73 ± 1,05       |
| Real-ESRGAN (FT)     | 362×362                  | 28,75 ± 3,33       | 0,76 ± 0,14        | 4,08 ± 0,58       |
| Real-ESRGAN (FT)     | 240×240                  | 28,43 ± 3,46       | 0,71 ± 0,14        | 2,53 ± 0,41       |
| HAT (FT)             | 240×240                  | 26,98 ± 3,04       | 0,68 ± 0,13        | 3,59 ± 0,62       |

##### Table 2 – Results in the reduced resolution domain (240×240 pixels)

| Method               | Training Resoluiton | PSNR (↑)           | SSIM (↑)           | PI (↓)            |
|----------------------|--------------------------|--------------------|--------------------|-------------------|
| FBP                  | –                        | 19,40 ± 1,89       | 0,47 ± 0,09        | 5,60 ± 1,72       |
| Real-ESRGAN (pre)    | –                        | 19,62 ± 2,13       | 0,57 ± 0,08        | 5,12 ± 1,46       |
| HAT (pre)            | –                        | 17,29 ± 2,30       | 0,41 ± 0,09        | 5,62 ± 2,00       |
| Real-ESRGAN (FT)     | 362×362                  | 28,99 ± 3,02       | 0,80 ± 0,11        | 4,88 ± 0,82       |
| Real-ESRGAN (FT)     | 240×240                  | 29,63 ± 3,44       | 0,81 ± 0,11        | 3,46 ± 0,72       |
| HAT (FT)             | 240×240                  | 27,67 ± 3,01       | 0,77 ± 0,10        | 4,86 ± 0,78       |

The boxplots below illustrate the distribution of the PSNR, SSIM, and PI metrics for the different evaluated methods, highlighting the gains achieved through fine-tuning and the reduction in result dispersion compared to direct inference.

##### Original Resolution (362×362 pixels)
![Boxplots - Resolução Original](/Others/Metrics/Results/boxplot_res_original.png)

##### Reduced Resoluiton (240×240 pixels)
![Boxplots - Resolução Reduzida](/Others/Metrics/Results/boxplot_res_reduzida.png)

The complete result files are available in the [`Results`](/Others/Metrics/Results/) directory, which contains both the boxplot figures and the file with the individual metrics computed for all reconstructed images in this experiment.

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
