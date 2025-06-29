# 🌧️ Rainfall Prediction Using Machine Learning

This project predicts whether it will rain or not based on weather features like temperature, humidity, wind, and cloud cover. Various classification algorithms are applied and evaluated.

---

## 📂 Project Overview

This machine learning model is built on a real-world rainfall dataset. The project includes:

- Exploratory Data Analysis (EDA)
- Preprocessing and Feature Scaling
- Handling class imbalance with `RandomOverSampler`
- Training multiple classification models
- Model comparison using accuracy and classification report

---

## 📊 Algorithms Used

- Logistic Regression
- Support Vector Classifier (SVC)
- XGBoost Classifier

---

## 📦 Libraries & Tools

- Python
- pandas, NumPy
- scikit-learn
- seaborn, matplotlib
- imbalanced-learn (`RandomOverSampler`)
- XGBoost
- Streamlit (for deployment)

---

## 📈 Workflow

1. Load and preprocess data (`Rainfall.csv`)
2. Scale features using `StandardScaler`
3. Split into training and test sets
4. Apply oversampling to balance classes
5. Train models and evaluate using accuracy and confusion matrix


