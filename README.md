# 🩺 Heart Failure Prediction using Machine Learning

> **Educational project:** This repository is for machine-learning practice and exploratory analysis. It is not a clinical diagnostic system and should not be used for medical decisions.

## 📌 Context & Motivation
Cardiovascular diseases (CVDs) are the **leading cause of death globally**. Early detection of heart conditions can **significantly improve treatment outcomes** and **save lives**. This project leverages **machine learning** to enable efficient and reliable **heart disease prediction**, aiding healthcare professionals in early diagnosis.

---

## ❓ Problem Statement
Heart disease often remains **undiagnosed** until it's too late, particularly in individuals with risk factors like **high blood pressure**, **diabetes**, or **high cholesterol**. The aim is to create a predictive model that can analyze patient health data and **accurately detect the likelihood of heart disease** before it becomes critical.

---

## 🎯 Project Aim
To develop a machine learning model that uses patient medical attributes to **predict heart disease**. This can assist medical professionals in identifying at-risk patients and **providing early interventions**.

---

## 🧾 Dataset Overview

| Feature           | Description |
|------------------|-------------|
| `Age`            | Age of the patient (in years) |
| `Sex`            | Biological sex (`M`: Male, `F`: Female) |
| `ChestPainType`  | Type of chest pain (`TA`, `ATA`, `NAP`, `ASY`) |
| `RestingBP`      | Resting blood pressure (mm Hg) |
| `Cholesterol`    | Serum cholesterol (mg/dl) |
| `FastingBS`      | Fasting blood sugar (`1` if > 120 mg/dl, else `0`) |
| `RestingECG`     | Resting electrocardiogram results (`Normal`, `ST`, `LVH`) |
| `MaxHR`          | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (`Y`/`N`) |
| `Oldpeak`        | ST depression induced by exercise |
| `ST_Slope`       | Slope of peak exercise ST segment (`Up`, `Flat`, `Down`) |
| `HeartDisease`   | Target variable (`1`: disease, `0`: normal) |

---

## 📒 Project Workflow

### 1. Data Loading and Initial Exploration
- Import libraries
- Load dataset into `heart_failure_df`
- Basic exploration and structure analysis

### 2. Preliminary Visualization & Feature Selection
- Visual distribution
- Correlation and medical relevance review

### 3. Data Cleaning
- Handle missing values & duplicates
- Detect & treat outliers
- Binning & smoothing

### 4. Feature Engineering
- Post-cleaning selection: Correlation matrix, Chi-square test, domain logic
- Normalization vs. Standardization using QQ Plot

### 5. Feature Encoding
- One-Hot Encoding for categorical and binned features

### 6. Data Splitting & Scaling
- Train-test split
- Apply normalization

---

## 📉 Dimensionality Reduction & Clustering

### 7. PCA (Principal Component Analysis)
- 2D PCA for visualization
- PCA retaining 95% variance for modeling

### 8. Clustering (Unsupervised Learning)

#### K-Means
- Elbow Method
- Dunn Index, Silhouette Score, Davies-Bouldin Index
- Cluster visualization

#### Hierarchical Clustering
- Dendrogram analysis
- Agglomerative clustering with optimal `k=2`

---

## 🤖 Classification Models

### 9. Model Training & Evaluation (With & Without PCA)
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVC)

Each model is evaluated using:
- Accuracy, Precision, Recall, F1-Score
- Decision Boundary visualization

### 10. Model Comparison
- Side-by-side performance analysis

---

## 🔧 Hyperparameter Tuning

### 11. Optimization Process
- K-Fold Cross-Validation
- `GridSearchCV` for hyperparameters
- Training & re-evaluation of:
  - Logistic Regression
  - Random Forest
  - SVC

### 12. Final Comparison
- Default vs. Tuned Models (with & without PCA)
- Final Decision Boundaries
- Feature Importance plots

---

## 📊 Results Summary
- Strong performance improvements from feature engineering and PCA
- Meaningful insights from clustering
- Optimized models outperform baselines
- Important medical features identified through feature importance analysis

---

## 🛠️ Technologies Used
- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **Jupyter Notebooks**

---

## 📊 Model Comparison

The notebook evaluates the models on a held-out test set of 184 samples. The strongest reported configuration by accuracy and F1-score is tuned Random Forest with PCA.

| Configuration | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 89.67% | 91.92% | 89.22% | 90.55% |
| Random Forest | 84.24% | 87.63% | 83.33% | 85.43% |
| Linear SVC | 89.67% | 91.09% | 90.20% | 90.64% |
| Tuned Linear SVC | 90.22% | 92.00% | 90.20% | 91.09% |
| Tuned Random Forest + PCA | **90.22%** | 89.62% | **93.14%** | **91.35%** |

**Selected model:** Tuned Random Forest with PCA, because it reports the highest recall and F1-score among the evaluated configurations while matching the best accuracy.

These results are from the notebook’s documented train/test split and should be interpreted as educational model evaluation, not clinical validation.

## 📁 Project Structure
heart-failure-prediction/
│
├── data/ # Dataset files
├── notebooks/ # Jupyter Notebooks for EDA and modeling
├── models/ # Saved models (if applicable)
├── outputs/ # Plots and reports
├── README.md # Project documentation
└── requirements.txt # Python dependencies

---

## ✅ Conclusion

This project demonstrates an end-to-end educational machine-learning workflow: preprocessing, feature engineering, dimensionality reduction, clustering, classification, and hyperparameter tuning. The reported metrics describe performance on this dataset and split only; the project is not a clinical diagnostic system and does not establish medical effectiveness.
