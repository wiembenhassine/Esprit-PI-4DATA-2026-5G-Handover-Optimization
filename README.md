# 5G Handover Optimization using Machine Learning

## Overview
This project was developed as part of the PI – 4th Year Data Program 
at **Esprit School of Engineering** (Academic Year 2025–2026).

It consists of a predictive machine learning pipeline to optimize 5G 
handover decisions using the DoNext dataset, following the CRISP-DM 
methodology.

## Features
- Early Warning system for QoS interruption risk (DSO1)
- Binary classification: Good QoS (≤50ms) vs Bad QoS (>50ms)
- Comparison of 3 models: Random Forest, XGBoost, GRU
- Full CRISP-DM pipeline (Phases 1 to 5)
- Exploratory Data Analysis across 3 scenarios (H-Bahn, Mobile, Static)

## Tech Stack

### Machine Learning
- Python 3.x
- Scikit-learn (Random Forest)
- XGBoost
- TensorFlow / Keras (GRU)

### Data Processing
- Pandas, NumPy
- Matplotlib, Seaborn

## Architecture
CRISP-DM Pipeline:
1. Business Understanding → DSOs definition
2. Data Understanding → EDA on DoNext dataset (~42.9M records)
3. Data Preparation → Cleaning, feature selection, standardization
4. Modeling → Random Forest + XGBoost + GRU (DSO1)
5. Evaluation → Accuracy, F1-Score, ROC-AUC

## Contributors
| Name | Role |
|------|------|
| Zakaria Zarrouk | Modeling & Pipeline |
| Mohamed Aziz | EDA & Data Understanding |
| Hamza Rezgui | Data Preparation |
| Hajer Maatoug | Evaluation & Metrics |
| Ghaya Korbi | Report & Documentation |
| Wiem Ben Hassine | Presentation & MLOps |

## Academic Context
Developed at **Esprit School of Engineering – Tunisia**  
PI – 4ème Data | Academic Year 2025–2026  
Supervisor: Rahma Bouraoui

## Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow
