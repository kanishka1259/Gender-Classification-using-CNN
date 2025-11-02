# CNN Gender Classification Project

## Overview  
This project is a hands-on implementation of a **Convolutional Neural Network (CNN)** that classifies gender (male/female) based on facial images. The work is inspired by and adapted from the Kaggle notebook: [*Gender Classification CNN*](https://www.kaggle.com/code/attanmhd/gender-classification-cnn/notebook).

## Purpose  
- To **learn** the workflow of image classification using CNNs: data preprocessing, model architecture, training, evaluation.  
- To build a practical project for portfolio use (GitHub, resume) demonstrating TensorFlow/Keras, computer vision, and deep learning skills.  
- To explore how hyperparameter tuning and data augmentation can affect performance in gender classification tasks.

## Dataset  
The dataset used for this project is available via Kaggle (link: https://www.kaggle.com/datasets/yasserhessein/gender-dataset)
**Note:** The actual dataset files are *not included* in this repository (see dataset usage instructions below).  


## Model & Methods  
- Built using **TensorFlow/Keras** in Python.  
- Architected a CNN with multiple convolutional + pooling layers, followed by fully connected layers.  
- Applied **data augmentation** (e.g., flips, rotations) to improve generalisation.  
- Monitored **accuracy** and **loss curves** during training to avoid overfitting.
  
## Results
- Achieved **~96.9% validation accuracy**.
- Included loss and accuracy plots in the `results/` directory for visualization.

## How to Run  
1. Clone or download this repository.  
   ```bash
   git clone https://github.com/your-github-username/cnn-gender-classification.git
