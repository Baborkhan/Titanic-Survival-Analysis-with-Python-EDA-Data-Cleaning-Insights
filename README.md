# 🚢 Titanic Survival Analysis — Complete Data Science Pipeline
### EDA · Machine Learning · Hyperparameter Tuning · Interactive Dashboard · Kaggle Submission

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.13+-purple?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?logo=kaggle)

---

## 📌 Overview

A **research-grade, end-to-end Data Science project** on the RMS Titanic passenger dataset — from raw data exploration to a fully deployed interactive ML dashboard.

This project is divided into **3 research notebooks** + **1 Streamlit web app**, covering every stage of a real-world data science workflow:

| Stage | Deliverable | Description |
|-------|-------------|-------------|
| **01** | `titanic_eda.ipynb` | Exploratory Data Analysis — 14 sections, 12 visualizations, hypothesis testing |
| **02** | `titanic_ml_pipeline.ipynb` | ML Pipeline — 7 models, 5-fold CV, SHAP-like explainability, learning curves |
| **03** | `titanic_hypertuning_kaggle.ipynb` | Hyperparameter Tuning + Kaggle `submission.csv` |
| **04** | `ml_app.py` | Interactive Streamlit Dashboard — live predictions, ROC curves, feature importance |

> **Author:** Ahsanul Yamin Babor · [Kaggle](https://www.kaggle.com/ahsanulyaminbabor) · [GitHub](https://github.com/Baborkhan)

---

## 📁 Project Structure

```
titanic-eda/
│
├── titanic_eda.ipynb                ← Notebook 1: Research-grade EDA (14 sections)
├── titanic_ml_pipeline.ipynb        ← Notebook 2: Full ML pipeline (19 sections)
├── titanic_hypertuning_kaggle.ipynb ← Notebook 3: Hypertuning + Kaggle submission
├── ml_app.py                        ← Streamlit interactive dashboard
│
├── data/
│   └── titanic.csv                  ← Raw dataset (download separately from Kaggle)
│
├── plots/                           ← Auto-generated visualizations (12 files)
│   ├── 01_data_quality.png
│   ├── 02_survival_overview.png
│   └── ...
│
├── models/                          ← Exported trained models (.pkl) — auto-created
├── output/                          ← submission.csv output — auto-created
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Notebook 1 — Exploratory Data Analysis

**14-section research pipeline:**

```
01  Environment Setup & Configuration
02  Data Acquisition & Inspection
03  Data Quality Assessment          → Missing value audit + heatmap
04  Data Cleaning & Feature Engineering → 8 new features
05  Descriptive Statistics           → Extended stats + frequency tables
06  Univariate — Categorical         → Count + proportion for all features
07  Univariate — Numerical           → Histogram + KDE + Boxplot + Q-Q plot
08  Bivariate Analysis               → Survival rate per feature
09  Multivariate Analysis            → Gender×Class heatmap + FacetGrid
10  Hypothesis Testing               → Chi-square, Mann-Whitney U, Point-Biserial
11  Outlier Analysis                 → IQR method for all numerical features
12  Correlation Analysis             → Full heatmap + feature-survival ranking
13  Research Findings                → 8 data-driven conclusions
14  ML Preprocessing Roadmap        → Action plan for model development
```

### Key Findings

| Finding | Detail |
|---------|--------|
| **Gender** | Female: 74% survived · Male: 19% — strongest predictor (χ²=260.7, p<0.001) |
| **Class** | 1st: 63% · 2nd: 47% · 3rd: 24% survived (χ²=102.9, p<0.001) |
| **Fare** | Survivors paid significantly higher fares; skewness=4.79 → requires log-transform |
| **Age** | Children (0–12): 58% · Seniors (60+): 27% · 19.9% missing values |
| **Family Size** | Small families (2–4) highest survival — non-linear relationship |
| **Has_Cabin** | Binary flag: 67% vs 30% survival — strong signal despite 77% raw missing |

### Statistical Tests

| Test | Variable | Result |
|------|----------|--------|
| Chi-Square | Sex vs Survived | χ²=260.7, p < 0.001 *** |
| Chi-Square | Pclass vs Survived | χ²=102.9, p < 0.001 *** |
| Chi-Square | Embarked vs Survived | χ²=26.5, p < 0.001 *** |
| Mann-Whitney U | Age: Survived vs Not | p < 0.05 * |
| Mann-Whitney U | Fare: Survived vs Not | p < 0.001 *** |
| Point-Biserial r | Fare ~ Survived | r=0.257, p < 0.001 *** |

---

## 🤖 Notebook 2 — Full ML Pipeline

**7 classifiers with 5-Fold Stratified Cross-Validation:**

- Logistic Regression
- SVM (RBF Kernel)
- Random Forest
- Extra Trees
- Gradient Boosting
- Decision Tree
- Voting Ensemble (LR + RF + GB)

**19 sections including:**
- 14 engineered features (Title, FamilySize, IsAlone, FarePerPerson, AgeGroup, etc.)
- Feature Importance (tree-based) + Permutation Importance (model-agnostic)
- SHAP-like local explanations for individual passengers
- Learning curves — bias-variance diagnosis for all 7 models
- Decision Tree visualization (depth-4, interpretable)
- Population-level feature contributions (200-sample analysis)

### Typical Model Rankings (5-Fold CV ROC-AUC)

| Rank | Model | ROC-AUC |
|------|-------|---------|
| 🥇 1 | Gradient Boosting | ~0.87 |
| 🥈 2 | Voting Ensemble | ~0.87 |
| 🥉 3 | Random Forest | ~0.86 |
| 4 | Extra Trees | ~0.85 |
| 5 | SVM | ~0.84 |
| 6 | Logistic Regression | ~0.83 |
| 7 | Decision Tree | ~0.79 |

---

## 🔧 Notebook 3 — Hyperparameter Tuning + Kaggle

- **GridSearchCV**: Logistic Regression, Decision Tree, SVM
- **RandomizedSearchCV**: Random Forest, Gradient Boosting, Extra Trees (n_iter=50–60)
- Tuned vs Baseline comparison with full visualization
- Voting Ensemble from top-3 tuned models
- Kaggle test set prediction → `output/submission.csv` (418 passengers)
- Full model export → `models/tuned_bundle.pkl`

---

## 🖥️ Streamlit Dashboard

Run: `streamlit run ml_app.py`

| Page | Content |
|------|---------|
| 🏠 Home | KPI cards, model leaderboard, dataset overview |
| 🤖 Model Comparison | ROC-AUC, F1, Accuracy for all 7 models |
| 📈 ROC & PR Curves | Full ROC + Precision-Recall curves |
| 🔍 Feature Importance | Tree-based + permutation importance |
| 📉 Learning Curves | Bias-variance plots for all 7 models |
| 🧩 Confusion Matrices | All 7 models, adjustable threshold |
| 🔮 Live Prediction | Input passenger → survival probability |
| 🧠 Explainability | Individual + population SHAP-like analysis |
| 🔧 Hyperparameter Tuning | Interactive GridSearch/RandomizedSearch |
| 🏅 Kaggle Submission | Upload test.csv → download submission.csv |
| 💾 Export Models | Download trained models as .pkl |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone
git clone https://github.com/Baborkhan/Titanic-Survival-Analysis-with-Python-EDA-Data-Cleaning-Insights.git
cd Titanic-Survival-Analysis-with-Python-EDA-Data-Cleaning-Insights

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from https://www.kaggle.com/c/titanic/data
#    Rename train.csv → titanic.csv, place in data/

# 4. Run notebooks in order
jupyter notebook

# 5. Launch dashboard
streamlit run ml_app.py
```

---

## 🧰 Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10 | Core language |
| Pandas | 1.5+ | Data manipulation |
| NumPy | 1.23+ | Numerical computation |
| Scikit-learn | 1.2+ | ML models, CV, metrics |
| SciPy | 1.9+ | Hypothesis testing |
| Matplotlib | 3.6+ | Base visualization |
| Seaborn | 0.12+ | Statistical plots |
| Plotly | 5.13+ | Interactive charts |
| Streamlit | 1.20+ | Web dashboard |

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

**Ahsanul Yamin Babor**

[![Kaggle](https://img.shields.io/badge/Kaggle-ahsanulyaminbabor-20BEFF?logo=kaggle)](https://www.kaggle.com/ahsanulyaminbabor)
[![GitHub](https://img.shields.io/badge/GitHub-Baborkhan-181717?logo=github)](https://github.com/Baborkhan)

---

*⭐ If this project helped you, please give it a star on GitHub!*
