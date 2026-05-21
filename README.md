# 🚢 Titanic — Advanced Survival Analysis & Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006600?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-2B9E2B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Deep-Dive EDA · 9 ML Models · XGBoost · LightGBM · Stacking · SHAP · 20 Engineered Features**

[📓 View Notebook](#notebook-overview) · [📊 Key Results](#model-performance) · [🚀 Quick Start](#quick-start) · [📁 Project Structure](#project-structure) · [🔗 Kaggle Profile](https://www.kaggle.com/baborkhan)

</div>

---

## 📌 Project Overview

A **production-grade** Titanic survival analysis that goes well beyond basic EDA. This notebook covers the full data science pipeline — from raw data audit to ensemble stacking — using industry-standard tools and rigorous cross-validation.

| Property | Detail |
|---|---|
| **Dataset** | 891 passengers · 12 raw features → **20 engineered features** |
| **Models** | XGBoost · LightGBM · Random Forest · Extra Trees · GBM · Logistic Reg · SVM · KNN · **Stacking** |
| **Validation** | **10-Fold Stratified CV** · Calibration · Threshold Tuning |
| **Best AUC** | **80.00%** (Logistic Regression, well-regularized) |
| **New vs Baseline** | +XGBoost +LightGBM +KNN +Stacking +RobustScaler +Interaction Features |

---

## 🎯 Key Findings

> **"Women survived 3× more than men (74% vs 19%). First-class passengers had 63% survival vs 24% for third class. Gender is the single strongest predictor of survival."**

- 🔵 **Sex** is the #1 predictor — female survival rate 74% vs male 19%
- 🟡 **Passenger Class** strongly separates outcomes (1st: 63% · 2nd: 47% · 3rd: 24%)
- 🟢 **Age** matters: children (<12) had the highest survival rate
- 🔴 **Family size** has a non-linear effect — solo travelers and large families both fared worse
- 🟣 **Cabin/Deck** holders survived at 67% vs 30% without cabin
- ⚪ **Fare** is strongly correlated (higher fare → higher class → better survival)

---

## 📊 Model Performance

### 10-Fold Stratified Cross-Validation Results

| # | Model | ROC-AUC | F1-Score | Accuracy | Train-CV Gap |
|---|---|---|---|---|---|
| 🥇 | **Logistic Regression** | **80.00%** | **66.7%** | 71.7% | 2% ← Low overfit |
| 🥈 | Extra Trees | 78.14% | 62.0% | 71.5% | 19% |
| 🥉 | Stacking Ensemble | 77.98% | 65.0% | 73.0% | 1% |
| 4 | Random Forest | 77.82% | 61.9% | 73.9% | 17% |
| 5 | SVM | 76.73% | 64.7% | 72.1% | 14% |
| 6 | Gradient Boost | 76.36% | 58.8% | 72.7% | 11% |
| 7 | XGBoost | 76.05% | 59.3% | 73.6% | 14% |
| 8 | LightGBM | 74.80% | 59.3% | 73.1% | 12% |
| 9 | KNN | 69.01% | 51.9% | 68.9% | 22% |

> **Note:** Logistic Regression wins because of its extremely low bias-variance gap (2%). Tree-based models overfit more on this small dataset.

---

## 🔬 Notebook Overview

```
01  Environment Setup & Global Style
02  Data Load & Audit
03  Advanced Feature Engineering (20 features)
04  Preprocessing — RobustScaler + Stratified Age Imputation
05  Exploratory Data Analysis (12 visualizations)
06  Statistical Hypothesis Testing (Chi-Square + Mann-Whitney U)
07  Correlation & Mutual Information Analysis
08  Model Definitions — 9 Classifiers (tuned hyperparameters)
09  10-Fold Stratified Cross-Validation Results
10  Model Comparison Visualizations
11  ROC Curves — All 9 Models
12  Confusion Matrices — Top 3 Models
13  Learning Curves — Bias-Variance Diagnosis
14  Permutation Feature Importance
15  Threshold Optimization & Precision-Recall
16  Model Calibration Analysis
17  Final Summary & Insights
```

---

## ⚙️ Feature Engineering (20 Features)

### From 12 Raw → 20 Engineered

| Feature | Description | Type |
|---|---|---|
| `Title_Enc` | Extracted from Name (Mr/Mrs/Miss/Master/Rare) | Categorical |
| `Family_Size` | SibSp + Parch + 1 | Numeric |
| `Is_Alone` | Family_Size == 1 | Binary |
| `Family_Bracket` | Small/Medium/Large family group | Ordinal |
| `Has_Cabin` | Cabin is known | Binary |
| `Deck_Enc` | Deck letter A–G extracted from Cabin | Ordinal |
| `Log_Fare` | log1p(Fare) — reduces right skew | Numeric |
| `Fare_Per_Person` | Fare / Family_Size | Numeric |
| `Fare_Bin` | Quartile-based fare bin | Ordinal |
| `Sex_Enc` | Binary sex encoding | Binary |
| `Embarked_Enc` | Port encoding (S/C/Q) | Categorical |
| `Pclass_Sex` | **NEW** Interaction: class × gender | Interaction |
| `Fare_x_Pclass` | **NEW** Log_Fare × Pclass | Interaction |
| `Is_Child` | Age < 12 | Binary |
| `Is_Senior` | Age > 60 | Binary |
| `Age_x_Pclass` | **NEW** Age × Pclass (post-imputation) | Interaction |

### Preprocessing
- **RobustScaler** (median centering + IQR scaling) — handles outliers better than StandardScaler
- **Stratified median imputation** for Age: grouped by Pclass × Sex to avoid information leakage

---

## 🚀 Quick Start

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm scipy jupyter
```

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Baborkhan/Titanic-Survival-Analysis-with-Python-EDA-Data-Cleaning-Insights.git
cd Titanic-Survival-Analysis-with-Python-EDA-Data-Cleaning-Insights

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook titanic_advanced_analysis.ipynb
```

### Run on Kaggle

1. Go to [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
2. Import this notebook
3. Change data path from `data/titanic.csv` → `/kaggle/input/titanic/train.csv`
4. Run all cells

---

## 📁 Project Structure

```
Titanic-Survival-Analysis/
│
├── titanic_advanced_analysis.ipynb   # Main notebook (35 cells, 17 sections)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── data/
    └── titanic.csv                   # Dataset (from Kaggle: titanic/train.csv)
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.0+ | Data manipulation |
| `numpy` | 1.24+ | Numerical computing |
| `matplotlib` | 3.7+ | Base plotting |
| `seaborn` | 0.12+ | Statistical visualization |
| `scikit-learn` | 1.3+ | ML models, CV, preprocessing |
| `xgboost` | 2.0+ | Gradient boosting |
| `lightgbm` | 4.0+ | Fast gradient boosting |
| `scipy` | 1.11+ | Statistical hypothesis tests |

---

## 📈 Statistical Hypothesis Testing

All features were validated with formal statistical tests:

| Feature | Test | Result | Effect Size |
|---|---|---|---|
| Sex | Chi-Square | p < 0.001 ★★★ | Cramér's V = 0.54 |
| Pclass | Chi-Square | p < 0.001 ★★★ | Cramér's V = 0.34 |
| Embarked | Chi-Square | p < 0.001 ★★★ | Cramér's V = 0.11 |
| Age | Mann-Whitney U | p < 0.001 ★★★ | r = 0.10 |
| Fare | Mann-Whitney U | p < 0.001 ★★★ | r = 0.26 |
| Family_Size | Mann-Whitney U | p = 0.015 ★ | r = 0.06 |

---

## 🔗 Related Links

- 📊 **Kaggle Profile**: [kaggle.com/baborkhan](https://www.kaggle.com/baborkhan)
- 🐙 **GitHub Profile**: [github.com/Baborkhan](https://github.com/Baborkhan)
- 🏆 **Titanic Competition**: [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- Inspired by the data science community's extensive Titanic analysis work
- Built with open-source Python ecosystem

---

<div align="center">

**⭐ If this notebook helped you, please give it a star!**

Made with ❤️ by [Baborkhan](https://github.com/Baborkhan)

</div>
