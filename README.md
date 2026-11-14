## 🚀 Live Demo

**Try the application here:** [Diabetes Prediction ML](https://diabetes-prediction-ml-mpgg9qe28x6dypfmfrpxo6.streamlit.app/)

An interactive machine learning application for diabetes risk prediction using multiple classification algorithms.

# Diabetes Prediction AI

An interactive machine learning application that predicts diabetes-positive or diabetes-negative classes using health measurements and a class-balanced Random Forest classifier.

## Project Overview

This project uses machine learning to classify diabetes outcomes based on eight input features. It includes model training, evaluation, visualization, and an interactive Streamlit dashboard.

**This project is for educational purposes only and is not a medical diagnostic tool.**

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Altair
- Joblib

## Dataset

Pima Indians Diabetes Dataset

- 768 records
- 8 input features
- 1 target variable

### Input Features

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## Machine Learning Models

Four algorithms were evaluated:

- Logistic Regression
- Random Forest
- Balanced Random Forest
- Support Vector Machine (SVM)

## Model Evaluation

The Balanced Random Forest achieved the following test results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 75.97% |
| Precision | 64.91% |
| Recall    | 68.52% |
| F1-Score  | 66.67% |
| ROC-AUC   | 0.8241 |

Five-fold cross-validation was also performed to evaluate model performance.

## Dashboard Features

- Interactive health measurement input form
- Diabetes class prediction
- Model confidence score
- Model performance metrics
- Comparison of four machine learning algorithms
- ROC curve comparison
- Classification report
- Confusion matrix visualization
- Feature importance visualization

## Project Structure

```text
Diabetes-Prediction-ML/
├── app.py
├── train.py
├── diabetes.csv
├── diabetes_model.pkl
├── model_evaluation.pkl
├── confusion_matrix.png
├── feature_importance.png
├── roc_curve.png
└── requirements.txt
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Diabetes-Prediction-ML
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the models

```bash
python train.py
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

## Disclaimer

This application is an experimental machine learning project. Its predictions are not medical diagnoses and should not be used to make healthcare decisions. Consult a qualified healthcare professional for medical assessment.
