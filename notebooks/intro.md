# Notebooks Workflow Guide

This directory contains the step-by-step implementation of the project, organized sequentially. Please run the notebooks in the following order to reproduce the results

### File Descriptions

**1. Data Acquisition**
* ' 01_data_query.ipynb'
* Connects to the Gaia DR3 Archive and cross match with other datasets using 'astroquery'.
* Downloads the required astrometric and photometric data.
* Saves the raw data locally.

**2. Preprocessing and Exploratory Data Analysis (EDA)**
*'02_preprocessing_and_EDA.ipynb'
* **Cleaning:** Handles missing values, outliers, and negative parallaxes
* **Feature Engineering:** Calculates Absolute Magnitude ($M_G$), Color index ($G_{BP}-G_{RP}$), and Tangential Velocity
* **EDA:** Visualizes the Hertzsprung-Russell (HR) Diagram, Galactic Maps, and Feature Correlations
* **Preparation:** Scales data (Quantile Transformer) and splits into Training/Test sets.

**3. Classification Task (Evolutionary Phases)**
* '03_model_classification.ipynb'
* Defines and trains two architectures: Pre-Norm ResNet MLP and Pre-Norm FT-Transformer.
* Objective: Classify stars into 5 phases (Main sequence, Sub-Giant, Giant, Supergiant, and White dwarf).
* Evaluation: Confusion Matrices, Accuracy, and F1-Score comparison.

**4. Regression Task (Stellar Mass & Age)**
* '04_model_regression.ipynb'
* Defines and trains the regression variation of the Pre-Norm ResNet MLP and Pre-Norm FT-Transformer.
* Objective: Predict fundamental parameters: Stellar mass and Age.
* Includes Warmup scheduler and custom R2-Scores metrices
* Evaluation: MAE, R2, and Physical Error Analysis (converting log predictions back to solar units/Gyr)

---
**Notes:** Ensure you have installed the dependencies listed in '../requirements.txt' before running these notebooks.
---
**Notes:** Ensure you have installed the dependencies listed in '../requeirements.txt' before running these notebooks.
