# Generative Models for Data Augmentation on Medical Data

## Overview

We evaluate the impact of data augmentation on the performance of a Convolutional Neural Network on 8 medical datasets ([MedMNIST](https://medmnist.com/) benchmark). We demonstrate accuracy improvements on 7 of the datasets when using either Geometric or Deep Learning data augmentation methods (Conditional VAE, Joint VAE, Conditional GAN).

## Datasets

We will look into applying the above project on some (if not all) of the following datasets from the MedMNIST Classification benchmark (https://medmnist.github.io/):

| Dataset | Classification Type | Train size | Validation size | Test size |
| :--- | :--- | :---: | :---: | :---: |  
| PathMNIST Pathology | Multi-Class (9) | 89,996 | 10,004 | 7,180 |
| DermaMNIST Dermatoscope | Multi-Class (7) | 7,007 | 1,003 | 2,005 |
| OCTMNIST OCT | Multi-Class (4) | 97,477 | 10,832 | 1,000 |
| PneumoniaMNIST Chest X-ray | Binary-Class (2) |4,708 | 524 | 624 |
| BreastMNIST Breast Ultrasound | Binary-Class (2) | 546 | 78 | 156 |
| OrganMNIST_Axial Abdominal CT | Multi-Class (11) | 34,581 | 6,491 | 17,778 |
| OragnMNIST_Coronal Abdominal CT | Multi-Class (11) |13,000 | 2,392 | 8,268 |
| OrganMNIST_Sagittal Abdominal CT | Multi-Class (11) | 13,940 | 2,452 | 8,829 |

## Google Colab

The code is available as a self-hosted, stand-alone Google Colab Notebook accessible [here](https://colab.research.google.com/drive/1J64flVq0ALWS7JBd8hj5qF1bwdbHlmeR?usp=sharing).