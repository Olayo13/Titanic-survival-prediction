# 🚢 Titanic Survival Prediction: A Machine Learning Approach

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📋 Project Overview
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. While there was some element of luck involved in surviving, it was not entirely random. This project builds a **Machine Learning pipeline** to predict which passengers survived the tragedy based on socio-economic and demographic data.

The goal was not just to achieve high accuracy, but to construct a **robust, leakage-free engineering workflow** that moves from raw data to a production-ready predictive model.

## ⚙️ Key Engineering Features
Unlike standard approaches, this project implements advanced preprocessing techniques to maximize information extraction:

* **Title-Based Age Imputation:** Instead of using a global average, missing ages were imputed using the median age of specific social titles (e.g., *Master*, *Mr*, *Miss*), preserving the demographic distribution.
* **Feature Engineering (FamilySize):** Combined `SibSp` (Siblings/Spouses) and `Parch` (Parents/Children) into a unified `FamilySize` variable. This revealed that small families (2-4 members) had a distinct survival advantage over solitary travelers or large groups.
* **Leakage-Free Pipeline:** The preprocessing logic was encapsulated to ensure that the test set was treated with the exact same rules derived from the training set, preventing data leakage.

## 🛠️ Methodology
The workflow follows a structured Data Science lifecycle:

1.  **Exploratory Data Analysis (EDA):** Analyzed correlations between Class, Gender, and Survival.
2.  **Data Cleaning:** Handled missing values in `Age`, `Embarked`, and `Fare`.
3.  **Feature Transformation:** - Extracted `Title` from passenger names.
    - Created `Has_Cabin` binary feature.
    - Encoded categorical variables using One-Hot Encoding.
4.  **Modeling:** Trained a **Random Forest Classifier** (`n_estimators=100`, `max_depth=5`) to capture non-linear relationships.
5.  **Validation:** Achieved an internal validation accuracy of **~82%**.

## 📊 Results
* **Model:** Random Forest Classifier
* **Internal Accuracy:** 82.1%
* **Key Predictors:** Gender, Title (Social Status), Fare (Class), and Family Size.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/tu_usuario/titanic-survival-prediction.git](https://github.com/tu_usuario/titanic-survival-prediction.git)
