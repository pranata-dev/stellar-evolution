**ACADEMIC INTEGRITY NOTICE**
This repository contains the source code for my ongoing undergraduate thesis at IPB University

**Status:** Work in Progress (WIP).
**Usage:** This code is published for portfolio and educational purpose only. Plagiarism and unauthorized use for academic submissions is strictly prohibited.
**Publication:** A Scientific paper based on this work is currently in preparation.

# Stellar Evolution Analysis using Pre-Norm FT-Transformer and Pre-Norm ResNet MLP model

## PROJECT OVERVIEW
This repository contains the code and resources for my Final Year Thesis.
The Project focuses on applying Neural Network architectures-specifically **Transformers** and **Multilayer Perceptrons** to study stellar evolutions.

Using astrometric and photometric data from **Gaia Data Release 3** This models perform two key tasks:
1. **Classifications:** Identifying the evolutionary phase of stars (e.g., Main Sequence, Sub-giant, Giant, Supergiant, and White Dwarf).
2. **Regressions:** Predicting fundamental stellar parameters: **Ages** and **Mass**.

## The Dataset: Gaia Data Release 3 & Cross-match with many literature.
Utilize high-precision data from the Gaia mission.
**Source:** Gaia Data Release 3 via Astroquery and crossmatch with data from Fusillo _et al._(2021) for White Dwarf stars and Pantaleoni _et al._(2021), Messineo _et al._(2019) and Zhang _et al._ (2020) for Supergiant stars.

**Input Features:** Parallax, BP-RP Color, Absolute Magnitude ($M_G$), Effective Temperature ($T_eff$),  etc.

**Preprocessing:** Data  cleaning, parallax inversion to distance, and normalizations.

## Model Architecture
I implemented and compared two distinct deep learning models:
1. **Pre-Norm FT-Transformer models:**
Using Pre-Norm FT Transformer, adapting the self-attention mechanism typically used in NLP for tabular physics data to capture complex non-linear
relationship between stellar parameters. Transform numerical features into embeddings (tokens) and processes them through Transformer blocks with Self-Attention.
*Optimization:* Implements **Pre-Normalization** (Pre-Norm) for  training stability and a Warmup learning rate schedule to prevent early divergence.
2. **Pre-Norm ResNet Multilayer Perceptron**
An enhanced Multilayer Perceptron inspired by ResNet (Residual Networks). Utilizes **Skip Connections** (Residual blocks) allowing the network to be much deeper without suffering
from the vanishing gradient problem.
*Optimization:* Uses Pre-Noralization layers before activation functions to smooth the loss landscape.

## Tech Stack
**Core Framework:** PyTorch.
**Libraries:** 'rtdl' (Revisiting Tabular Deep Learning) or Custom Implementation, Scikit-learn, Pandas.
**Visualization:** Matplotlib, Seaborn.
**Physics Tools:** Astropy, Astroquery

## Citation / Author
Created by **Dzulfikar Yudha Pranata** as part of the Physics Bachelor's Thesis at IPB University
* **Email:** [dzulfikaryudha@gmail.com]
* **LinkedIn:** [https://www.linkedin.com/in/pranatayudha26]
