# Handwritten Math Symbols Classification

## Overview
This project implements a convolutional neural network (CNN) using TensorFlow and Keras to classify handwritten math symbols. The model is trained to recognize symbols representing digits (0-9) and basic arithmetic operations (+, -, *, /). 

## Getting Started

### Dataset
The dataset consists of images organized into subfolders for each class:

- 0, 1, 2, 3, 4, 5, 6, 7, 8, 9: Digits
- add, div, eq, mul, sub: Arithmetic operators
- x, y, z: Other symbols

The dataset used in this project is sourced from Kaggle. You can find the dataset at the following link: [Handwritten Math Symbols Dataset](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols).

### Prerequisites
- Python 3
- TensorFlow
- Other dependencies (install using `pip install -r requirements.txt`)

### Installation
1. Clone the Repository:
   ```bash
   git clone https://github.com/your-username/handwritten-math-symbols.git
   cd handwritten-math-symbols
   ```
   Make sure to replace "your-username" with your actual GitHub username and update the "Author" section with your name.
  
3. Install Dependencies:
   ```bash
   pip install -r requirements.txt

### Usage
1. Organize Dataset:
   Update the `dataset_dir` variable in code.py with the path to your directory.
  
2. Run the Code:
   ```bash
   python code.py
   ```

### Results
The trained model achieved the following results on the test set:

**Test Loss: 0.2166**

**Test Accuracy: 93.15%**

**Test AUC: 0.9960**
