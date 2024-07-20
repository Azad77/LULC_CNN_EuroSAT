# LULC_CNN_EuroSAT

This repository contains code for implementing a Convolutional Neural Network (CNN) for land use and land cover classification using the EuroSAT dataset and PyTorch. The EuroSAT dataset is based on Sentinel-2 satellite images and includes 10 classes representing various land use and land cover types.


## Project Overview

This project demonstrates how to use a CNN for image classification tasks with the EuroSAT dataset. The implementation includes:
- Data preprocessing and augmentation
- Building and training a CNN model
- Evaluating model performance
- Making predictions on sample images

The CNN model used is based on the ResNet-50 architecture, and the code covers the full workflow from data loading to model evaluation.

## Prerequisites

Before running the code, ensure you have the following libraries installed:
- `torch` and `torchvision` for deep learning
- `PIL` for image processing
- `matplotlib`, `seaborn`, and `pandas` for data visualization and manipulation
- `scikit-learn` for evaluation metrics

You can install these libraries using pip:

```bash
pip install torch torchvision pillow matplotlib seaborn pandas scikit-learn
```
Installation
Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/LULC_CNN_EuroSAT.git
```

Navigate to the project directory:

```bash
cd LULC_CNN_EuroSAT
```

Usage
Download and Unzip the EuroSAT Dataset:

Download the EuroSAT dataset and unzip it into the ./EuroSAT/ directory. You can use the following commands:

```bash
wget http://madm.dfki.de/files/sentinel/EuroSAT.zip -O EuroSAT.zip
unzip -q EuroSAT.zip -d 'EuroSAT/'
rm EuroSAT.zip
```

References

Reid Falconer, Land Use and Land Cover Classification (Beating the Benchmark). Available at: GitHub Repository

Helber, P., Bischke, B., Dengel, A., & Borth, D. (2018). Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. arXiv preprint arXiv:1709.00029. Available at: arXiv

Ankur Mahesh & Isabelle Tingzon, Land Use and Land Cover Classification using PyTorch. Available at: Google Colab

License
This project is licensed under the MIT License - see the LICENSE file for details.
