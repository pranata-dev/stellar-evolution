# Notebooks Workflow Guide

This directory contains the step-by-step implementation of the project, organized sequentially. Please run the notebooks in the following order to reproduce the results

### File Descriptions

**1. Data Acquisition**
* ' 01_data_query.ipynb'
* Connects to the Gaia DR3 Archive and cross match with other datasets using 'astroquery'.
* Downloads the required astrometric and photometric data.
* Saves the raw data locally.

**2. Data Preparation**
* '02_preprocessing.ipynb'
* Handle missing values and outliers.
* Performs featureengineering (calculating absolute magnitude, etc).
* Splits data into Training, Validation, and Test sets.
* Applies normalization/scaling appropriate for Neural Networks.

**3. Model Training (The Core)**
* ' 03_model_mlp.ipynb'
* Implementation of the Pre-Norm ResNet MLP (Baseline model).
* Training loop with residual connections.
* ' 04_model_transformer.ipynb'
* Implementation of the Pre-Norm FT-Transformer (Feature Tokenizer) architecture
* Includes the training loop with Pre-Norm and Warmup scheduler

**4. Analysis**
* '05_evaluation_analysis.ipynb'
* Loads the trained models.
* Visualizes performance: Loss Curves, Confusion Matrices, and ROC Curves.
* Physics check: Reconstructs the Hertzsprung-Russell Diagram (HR Diagram) using predicted values vs actual Gaia Data

---
**Notes:** Ensure you have installed the dependencies listed in '../requeirements.txt' before running these notebooks.
