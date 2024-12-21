# Car or Truck Classification using ResNet50

This project is a deep learning pipeline for classifying images of cars and trucks using a pre-trained ResNet50 model fine-tuned on a custom dataset. It utilizes TensorFlow and Keras for model building and training, and includes data augmentation techniques to improve model performance.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Confusion Matrix](#confusion-matrix)
- [Results](#results)
- [License](#license)

## Overview

This project demonstrates how to:
- Download and preprocess the "Car or Truck" dataset.
- Use data augmentation techniques to improve model generalization.
- Fine-tune a pre-trained ResNet50 model for image classification.
- Evaluate the model's performance on a test set using metrics such as accuracy and confusion matrix.

## Dataset

The dataset is sourced from Kaggle's ["Car or Truck"](https://www.kaggle.com/ryanholbrook/car-or-truck) dataset, which contains images labeled as either cars or trucks. The data is split into 80% for training and 20% for testing.

## Requirements

To run the project, you need the following libraries:

- TensorFlow
- Keras
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- kagglehub

Install all dependencies using the following command:

```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn kagglehub
# car-or-truck
