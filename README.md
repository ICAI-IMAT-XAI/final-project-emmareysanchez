
# XAI for Fashion Image Classification
## Actionable Explainability Case Study with DeepFashion
This project studies the use of Explainable Artificial Intelligence (XAI) techniques to analyse and improve the behaviour of a deep learning model for fashion image classification. Using a subset of the DeepFashion dataset, the project combines local and global explanations with an explainability-driven intervention to reduce systematic classification errors.

## How to Run the Experiments

### 1. Dataset
Download the DeepFashion dataset from Kaggle:
https://www.kaggle.com/datasets/pawakapan/deepfashion
Extract the dataset locally and ensure the image paths are correctly set in make_splits.py.

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Create Train / Validation / Test Splits
python scripts/make_splits.py

This generates the processed dataset under:

data/processed/

### 4. Main Analysis and Explainability

Run the main notebook:

notebooks/deepfashion.ipynb

This notebook contains:
- Model training and evaluation
- Grad-CAM++ visualisations (local and global)
- Sanity checks for explanations

### 5. Actionable XAI Experiment

Run the actionable explainability experiment:

python scripts/actionable_xai.py

This script trains the BEFORE and AFTER models and saves the final comparison of confusion matrices.