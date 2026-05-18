"""
🚢 Titanic ML Pipeline — Complete Machine Learning App
6 Models + Full Preprocessing + Cross-Validation + ROC-AUC + Feature Importance + Live Prediction
Run: streamlit run ml_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle, io, warnings, time
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                      learning_curve, validation_curve)
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              confusion_matrix, roc_curve, precision_recall_curve,
                              classification_report, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚢 Titanic ML Pipeline",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0d14; }
  [data-testid="stSidebar"]          { background: #111827; border-right: 1px solid #1f2937; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  .kpi-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    border-radius: 14px;
    padding: 20px 12px 16px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
    transition: transform .2s;
  }
  .kpi-card:hover { transform: translateY(-2px); }
  .kpi-value { font-size: 2em; font-weight: 800; letter-spacing: -0.5px; }
  .kpi-label { font-size: 0.73em; color: #6b7280; margin-top: 4px;
                text-transform: uppercase; letter-spacing: 0.08em; }

  .section-header {
    font-size: 1.1em; font-weight: 700; color: #f9fafb;
    border-left: 4px solid #6366f1;
    padding: 6px 0 6px 14px;
    margin: 20px 0 14px 0;
    background: rgba(99,102,241,0.08);
    border-radius: 0 8px 8px 0;
  }

  .model-card {
    background: linear-gradient(135deg, #111827, #1a2236);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
  }
  .best-model {
    border: 1px solid #10b981;
    background: linear-gradient(135deg, #0d1f17, #122d22);
  }

  .insight { background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
    border-left:4px solid #10b981; border-radius:8px; padding:12px 16px;
    margin:6px 0; color:#d1fae5; font-size:0.88em; }
  .warning { background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
    border-left:4px solid #ef4444; border-radius:8px; padding:12px 16px;
    margin:6px 0; color:#fee2e2; font-size:0.88em; }
  .info { background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3);
    border-left:4px solid #6366f1; border-radius:8px; padding:12px 16px;
    margin:6px 0; color:#e0e7ff; font-size:0.88em; }
  .predict-survived {
    background: linear-gradient(135deg,#064e3b,#065f46);
    border:2px solid #10b981; border-radius:14px;
    padding:28px; text-align:center; margin:12px 0;
  }
  .predict-not {
    background: linear-gradient(135deg,#450a0a,#7f1d1d);
    border:2px solid #ef4444; border-radius:14px;
    padding:28px; text-align:center; margin:12px 0;
  }

  h1,h2,h3,h4,p,li { color: #f9fafb !important; }
  .stMarkdown p { color: #d1d5db !important; }
  .stDataFrame { background: #111827; }
  div[data-testid="stMetric"] { background: #111827; border-radius:10px; padding:12px; }
</style>
""", unsafe_allow_html=True)

# ─── Color Palette ────────────────────────────────────────────────────────────
C_GREEN  = '#10b981'
C_RED    = '#ef4444'
C_BLUE   = '#6366f1'
C_YELLOW = '#f59e0b'
C_CYAN   = '#06b6d4'
C_PINK   = '#ec4899'
C_PURPLE = '#8b5cf6'
C_ORANGE = '#f97316'

MODEL_COLORS = {
    'Logistic Regression': C_BLUE,
    'SVM':                 C_PINK,
    'Random Forest':       C_GREEN,
    'Extra Trees':         C_CYAN,
    'Gradient Boosting':   C_YELLOW,
    'Decision Tree':       C_ORANGE,
    'Voting Ensemble':     C_PURPLE,
}

def dark_fig(fig, height=None):
    upd = dict(
        paper_bgcolor='#111827', plot_bgcolor='#111827',
        font=dict(color='#f9fafb', size=11),
        xaxis=dict(gridcolor='#1f2937', linecolor='#374151'),
        yaxis=dict(gridcolor='#1f2937', linecolor='#374151'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#f9fafb')),
        margin=dict(l=50, r=30, t=50, b=40)
    )
    if height: upd['height'] = height
    fig.update_layout(**upd)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv('data/titanic.csv')
        return df.drop_duplicates()
    except FileNotFoundError:
        pass
    try:
        import requests
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        r   = requests.get(url, timeout=10)
        df  = pd.read_csv(io.StringIO(r.text))
        return df.drop_duplicates()
    except Exception:
        pass
    try:
        import seaborn as sns
        df = sns.load_dataset('titanic')
        df = df.rename(columns={
            'survived': 'Survived', 'pclass': 'Pclass', 'sex': 'Sex',
            'age': 'Age', 'sibsp': 'SibSp', 'parch': 'Parch',
            'fare': 'Fare', 'embarked': 'Embarked', 'who': 'Name',
            'deck': 'Cabin', 'alive': '_alive'
        })
        df['Survived'] = df['Survived'].astype(int)
        df['Name'] = df.apply(
            lambda r: f"Passenger, {'Mr.' if r['Sex']=='male' else 'Mrs.'} Demo", axis=1)
        df['Ticket'] = '12345'
        df['PassengerId'] = range(1, len(df)+1)
        st.warning("⚠️ Using seaborn fallback dataset. For full accuracy, place real train.csv in data/titanic.csv")
        return df.drop_duplicates()
    except Exception as e:
        st.error(f"❌ Could not load any data source: {e}. Place titanic.csv in data/ folder.")
        st.stop()

@st.cache_data(show_spinner=False)
def engineer_features(df):
    d = df.copy()

    # Title extraction
    d['Title'] = d['Name'].str.extract(r',\s*([^.]+)\.')
    rare = d['Title'].value_counts()
    d['Title'] = d['Title'].replace(rare[rare < 10].index, 'Rare')
    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    d['Title_Enc'] = d['Title'].map(title_map).fillna(4).astype(int)

    # Family
    d['Family_Size']    = d['SibSp'] + d['Parch'] + 1
    d['Is_Alone']       = (d['Family_Size'] == 1).astype(int)
    d['Family_Cat_num'] = pd.cut(d['Family_Size'], bins=[0,1,4,20],
                                  labels=[0,1,2]).astype(float).fillna(0).astype(int)

    # Cabin / Fare
    d['Has_Cabin']      = d['Cabin'].notna().astype(int)
    d['Log_Fare']       = np.log1p(d['Fare'])
    d['Fare_Per_Person'] = d['Fare'] / d['Family_Size']

    # Encode Sex
    d['Sex_Enc'] = (d['Sex'] == 'female').astype(int)

    # Embarked
    d['Embarked'] = d['Embarked'].fillna('S')
    d['Embarked_Enc'] = d['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0).astype(int)

    # Deck
    d['Deck'] = d['Cabin'].str[0].fillna('U')

    # Age group
    d['Age_Group_num'] = pd.cut(d['Age'], bins=[0,12,18,35,60,120],
                                 labels=[0,1,2,3,4]).astype(float)

    return d

FEATURES = [
    'Pclass', 'Sex_Enc', 'Age', 'SibSp', 'Parch', 'Fare',
    'Log_Fare', 'Family_Size', 'Is_Alone', 'Has_Cabin',
    'Embarked_Enc', 'Title_Enc', 'Family_Cat_num', 'Fare_Per_Person'
]
FEATURE_LABELS = {
    'Pclass': 'Passenger Class', 'Sex_Enc': 'Sex (Female=1)', 'Age': 'Age',
    'SibSp': 'Siblings/Spouse', 'Parch': 'Parents/Children', 'Fare': 'Fare (£)',
    'Log_Fare': 'Log(Fare)', 'Family_Size': 'Family Size', 'Is_Alone': 'Is Alone',
    'Has_Cabin': 'Has Cabin', 'Embarked_Enc': 'Embarked Port',
    'Title_Enc': 'Title', 'Family_Cat_num': 'Family Category', 'Fare_Per_Person': 'Fare/Person'
}

@st.cache_data(show_spinner=False)
def prepare_xy(_df_eng):
    df = _df_eng.copy()
    X = df[FEATURES].copy()

    # Age: group-based imputation
    for pclass in [1, 2, 3]:
        for sex in ['male', 'female']:
            sex_enc = 1 if sex == 'female' else 0
            mask_all     = (df['Pclass'] == pclass) & (df['Sex'] == sex)
            mask_valid   = mask_all & df['Age'].notna()
            median_age   = df.loc[mask_valid, 'Age'].median()
            mask_missing = mask_all & df['Age'].isna()
            X.loc[mask_missing, 'Age'] = median_age

    X['Age']            = X['Age'].fillna(X['Age'].median())
    X['Fare_Per_Person'] = X['Fare_Per_Person'].fillna(X['Fare_Per_Person'].median())

    y = df['Survived'].copy()
    return X.astype(float), y

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_models():
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced',
                             solver='lbfgs', random_state=42)
    svm = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale',
               class_weight='balanced', random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=4,
                                 min_samples_leaf=2, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_split=4,
                               class_weight='balanced', random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, random_state=42)
    dt = DecisionTreeClassifier(max_depth=6, min_samples_split=4, min_samples_leaf=2,
                                 class_weight='balanced', random_state=42)
    ensemble = VotingClassifier(
        estimators=[('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42))],
        voting='soft'
    )
    return {
        'Logistic Regression': lr,
        'SVM':                 svm,
        'Random Forest':       rf,
        'Extra Trees':         et,
        'Gradient Boosting':   gb,
        'Decision Tree':       dt,
        'Voting Ensemble':     ensemble,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING (CACHED)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def run_full_pipeline(_X, _y):
    X, y = _X.values, _y.values

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models  = get_models()
    results = {}
    trained = {}

    for name, model in models.items():
        cv_res = cross_validate(
            model, X_sc, y, cv=cv,
            scoring=['accuracy', 'roc_auc', 'f1'],
            return_train_score=True,
            return_estimator=True
        )
        results[name] = {
            'acc_cv':  cv_res['test_accuracy'],
            'auc_cv':  cv_res['test_roc_auc'],
            'f1_cv':   cv_res['test_f1'],
            'train_acc_cv': cv_res['train_accuracy'],
            'train_auc_cv': cv_res['train_roc_auc'],
        }
        # Fit on full data for feature importance & predictions
        model.fit(X_sc, y)
        trained[name] = model

    # Best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]['auc_cv'].mean())

    # Full prediction probas (for ROC curves)
    probas = {}
    for name, model in trained.items():
        probas[name] = model.predict_proba(X_sc)[:, 1]

    # Feature importances (from tree-based models)
    feat_imp = {}
    for name in ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'Decision Tree']:
        if hasattr(trained[name], 'feature_importances_'):
            feat_imp[name] = trained[name].feature_importances_

    # LR coefficients
    feat_imp['Logistic Regression'] = np.abs(trained['Logistic Regression'].coef_[0])

    return {
        'results':   results,
        'trained':   trained,
        'scaler':    scaler,
        'X_sc':      X_sc,
        'y':         y,
        'probas':    probas,
        'feat_imp':  feat_imp,
        'best_name': best_name,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='text-align:center;padding:16px 0 8px'>
  <span style='font-size:2.6em'>🚢</span>
  <h2 style='color:#f9fafb;margin:6px 0 0;font-size:1.1em'>Titanic ML Pipeline</h2>
  <p style='color:#6b7280;font-size:0.72em;margin:2px 0 0'>6 Models · Full CV · Live Prediction</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("📑 Navigation", [
    "🏠 Home",
    "⚙️ Preprocessing",
    "🤖 Model Comparison",
    "📈 ROC & PR Curves",
    "🔍 Feature Importance",
    "📉 Learning Curves",
    "🧩 Confusion Matrices",
    "🔮 Live Prediction",
    "🧠 Explainability",
    "🔧 Hyperparameter Tuning",
    "🏅 Kaggle Submission",
    "💾 Export Models",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.78em;color:#6b7280;padding:8px 0'>
<b style='color:#9ca3af'>Models:</b><br>
  ① Logistic Regression<br>
  ② SVM (RBF Kernel)<br>
  ③ Random Forest<br>
  ④ Extra Trees<br>
  ⑤ Gradient Boosting<br>
  ⑥ Decision Tree<br>
  ⑦ Voting Ensemble (LR+RF+GB)
</div>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0a0d14,#111827,#1a1040);
     padding:28px 32px;border-radius:14px;margin-bottom:20px;
     border:1px solid #374151">
  <h1 style="color:white;margin:0;font-size:1.85em;font-family:Georgia,serif">
    🚢 Titanic ML Pipeline
  </h1>
  <p style="color:#6b7280;margin:6px 0 0;font-size:0.88em">
    Complete Machine Learning System &nbsp;·&nbsp;
    6 Models + Voting Ensemble &nbsp;·&nbsp;
    5-Fold Stratified CV &nbsp;·&nbsp; ROC-AUC · F1 · Feature Importance · Live Prediction
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Load & Train ─────────────────────────────────────────────────────────────
with st.spinner("⚙️ Loading data & training 7 models on 5-fold CV..."):
    raw_df  = load_data()
    eng_df  = engineer_features(raw_df)
    X, y    = prepare_xy(eng_df)
    pipe    = run_full_pipeline(X, y)

results   = pipe['results']
trained   = pipe['trained']
scaler    = pipe['scaler']
X_sc      = pipe['X_sc']
y_arr     = pipe['y']
probas    = pipe['probas']
feat_imp  = pipe['feat_imp']
best_name = pipe['best_name']

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    # KPI row
    cols = st.columns(6)
    total     = len(raw_df)
    survived  = int(raw_df['Survived'].sum())
    surv_rate = survived / total * 100
    n_feat    = len(FEATURES)
    n_models  = len(results)
    best_auc  = results[best_name]['auc_cv'].mean() * 100

    kpis = [
        ("Passengers", f"{total:,}",       "#6366f1", "👥"),
        ("Survived",   f"{survived:,}",    C_GREEN,   "✅"),
        ("Surv. Rate", f"{surv_rate:.1f}%",C_YELLOW,  "📊"),
        ("Features",   f"{n_feat}",        C_CYAN,    "🔧"),
        ("Models",     f"{n_models}",      C_PINK,    "🤖"),
        ("Best AUC",   f"{best_auc:.1f}%", C_GREEN,   "🏆"),
    ]
    for col, (label, val, color, icon) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi-card">
          <div style="font-size:1.4em">{icon}</div>
          <div class="kpi-value" style="color:{color}">{val}</div>
          <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.6, 1])

    with c1:
        st.markdown('<div class="section-header">📊 Model Leaderboard — 5-Fold CV Results</div>',
                    unsafe_allow_html=True)

        rows = []
        for name, res in results.items():
            rows.append({
                'Model':        name,
                'Accuracy':     f"{res['acc_cv'].mean()*100:.2f}% ± {res['acc_cv'].std()*100:.2f}",
                'ROC-AUC':      f"{res['auc_cv'].mean()*100:.2f}% ± {res['auc_cv'].std()*100:.2f}",
                'F1-Score':     f"{res['f1_cv'].mean()*100:.2f}% ± {res['f1_cv'].std()*100:.2f}",
                'AUC_sort':     res['auc_cv'].mean(),
                'Best':         '🏆' if name == best_name else '',
            })

        lb_df = pd.DataFrame(rows).sort_values('AUC_sort', ascending=False).drop('AUC_sort', axis=1)
        st.dataframe(lb_df, use_container_width=True, hide_index=True)

        # Bar comparison
        fig = go.Figure()
        metrics_to_show = [
            ('auc_cv',  'ROC-AUC',  0.8),
            ('acc_cv',  'Accuracy', 0.5),
            ('f1_cv',   'F1-Score', 0.3),
        ]
        names_sorted = sorted(results.keys(),
                              key=lambda k: results[k]['auc_cv'].mean(), reverse=True)
        colors_bar = [C_BLUE, C_CYAN, C_GREEN]
        for (metric, label, opacity), color in zip(metrics_to_show, colors_bar):
            vals = [results[n][metric].mean() for n in names_sorted]
            errs = [results[n][metric].std() for n in names_sorted]
            fig.add_trace(go.Bar(
                name=label, x=names_sorted, y=vals,
                error_y=dict(type='data', array=errs, visible=True),
                marker_color=color, opacity=0.9,
                text=[f"{v*100:.1f}%" for v in vals], textposition='outside',
            ))
        fig.update_layout(barmode='group', yaxis_range=[0, 1.08],
                          yaxis_tickformat='.0%',
                          title='Model Comparison — All Metrics (5-Fold CV)')
        dark_fig(fig, 460)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">🏆 Best Model</div>', unsafe_allow_html=True)
        best_res = results[best_name]
        st.markdown(f"""
        <div class="model-card best-model" style="text-align:center">
          <div style="font-size:2.2em">🥇</div>
          <div style="font-size:1.25em;font-weight:800;color:{C_GREEN};margin:8px 0">
            {best_name}
          </div>
          <div style="color:#6b7280;font-size:0.8em">Best by ROC-AUC</div>
          <hr style="border-color:#1f2937;margin:12px 0">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
            <div>
              <div style="font-size:1.5em;font-weight:700;color:{C_GREEN}">
                {best_res['auc_cv'].mean()*100:.2f}%
              </div>
              <div style="font-size:0.72em;color:#6b7280">ROC-AUC</div>
            </div>
            <div>
              <div style="font-size:1.5em;font-weight:700;color:{C_BLUE}">
                {best_res['acc_cv'].mean()*100:.2f}%
              </div>
              <div style="font-size:0.72em;color:#6b7280">Accuracy</div>
            </div>
            <div>
              <div style="font-size:1.5em;font-weight:700;color:{C_YELLOW}">
                {best_res['f1_cv'].mean()*100:.2f}%
              </div>
              <div style="font-size:0.72em;color:#6b7280">F1-Score</div>
            </div>
            <div>
              <div style="font-size:1.5em;font-weight:700;color:{C_PINK}">
                {best_res['auc_cv'].std()*100:.2f}%
              </div>
              <div style="font-size:0.72em;color:#6b7280">AUC Std Dev</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📌 Pipeline Summary</div>', unsafe_allow_html=True)
        pipeline_steps = [
            ("Data Source", "Titanic manifest · 891 records"),
            ("Missing: Age", "Group-based median (Pclass×Sex)"),
            ("Missing: Cabin", "Drop raw → Has_Cabin flag"),
            ("Missing: Embarked", "Mode imputation"),
            ("Encoding", "Sex, Embarked, Title → numeric"),
            ("New Features", "Log_Fare, Family_Size, Is_Alone,\nFare_Per_Person, Family_Cat"),
            ("Scaling", "StandardScaler"),
            ("Imbalance", "class_weight='balanced'"),
            ("Validation", "5-Fold Stratified CV"),
            ("Metrics", "ROC-AUC, F1, Accuracy"),
        ]
        for step, detail in pipeline_steps:
            st.markdown(f"""
            <div style="display:flex;gap:12px;padding:5px 0;border-bottom:1px solid #1f2937">
              <div style="color:#6366f1;font-weight:600;min-width:140px;font-size:0.82em">{step}</div>
              <div style="color:#d1d5db;font-size:0.82em">{detail}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Preprocessing":
    st.markdown("## ⚙️ Preprocessing Pipeline")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Missing Value Analysis</div>',
                    unsafe_allow_html=True)
        miss = raw_df.isnull().sum().reset_index()
        miss.columns = ['Feature', 'Missing']
        miss['%'] = (miss['Missing'] / len(raw_df) * 100).round(2)
        miss = miss[miss['Missing'] > 0].sort_values('%', ascending=False)

        fig = px.bar(miss, x='Feature', y='%', text='%',
                     color='%', color_continuous_scale='RdYlGn_r',
                     title='Missing Values (%)')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 90])
        dark_fig(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Class Distribution (Imbalance Check)</div>',
                    unsafe_allow_html=True)
        vc = raw_df['Survived'].value_counts()
        labels_map = {0: 'Not Survived', 1: 'Survived'}
        fig2 = px.pie(values=vc.values, names=[labels_map[i] for i in vc.index],
                      color=[labels_map[i] for i in vc.index],
                      color_discrete_map={'Survived': C_GREEN, 'Not Survived': C_RED},
                      hole=0.55, title='Target Class Balance')
        dark_fig(fig2, 360)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Engineered Features — Statistics</div>',
                unsafe_allow_html=True)
    feat_df = X.copy()
    feat_df.columns = [FEATURE_LABELS.get(c, c) for c in feat_df.columns]
    st.dataframe(feat_df.describe().round(3), use_container_width=True)

    st.markdown('<div class="section-header">Age Imputation — Before vs After</div>',
                unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(raw_df, x='Age', nbins=40,
                            color_discrete_sequence=[C_RED],
                            title=f'Before: {raw_df["Age"].isna().sum()} missing values')
        dark_fig(fig3, 320)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = px.histogram(X, x='Age', nbins=40,
                            color_discrete_sequence=[C_GREEN],
                            title='After: 0 missing (group-based median imputation)')
        dark_fig(fig4, 320)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">Feature Correlation Heatmap</div>',
                unsafe_allow_html=True)
    corr = X.copy()
    corr.columns = [FEATURE_LABELS.get(c, c) for c in corr.columns]
    corr_matrix = corr.corr().round(3)
    fig5 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, aspect='auto', height=520)
    fig5.update_traces(texttemplate='%{z:.2f}', textfont_size=8)
    dark_fig(fig5)
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown("## 🤖 Model-by-Model Cross-Validation Analysis")

    for name, res in sorted(results.items(),
                             key=lambda kv: kv[1]['auc_cv'].mean(), reverse=True):
        is_best = name == best_name
        card_cls = "model-card best-model" if is_best else "model-card"
        color = MODEL_COLORS[name]

        with st.expander(
            f"{'🏆 ' if is_best else ''}{name}  —  "
            f"AUC {res['auc_cv'].mean()*100:.2f}%  |  "
            f"Acc {res['acc_cv'].mean()*100:.2f}%  |  "
            f"F1 {res['f1_cv'].mean()*100:.2f}%",
            expanded=is_best
        ):
            c1, c2, c3 = st.columns(3)
            for col, (metric, label, fold_vals) in zip(
                [c1, c2, c3],
                [('auc_cv', 'ROC-AUC', res['auc_cv']),
                 ('acc_cv', 'Accuracy', res['acc_cv']),
                 ('f1_cv',  'F1-Score', res['f1_cv'])]
            ):
                fig = go.Figure()
                fold_nums = [f"Fold {i+1}" for i in range(len(fold_vals))]
                train_vals = res[f'train_{metric}'] if f'train_{metric}' in res else [None]*5

                if train_vals[0] is not None:
                    fig.add_trace(go.Scatter(
                        x=fold_nums, y=train_vals, mode='lines+markers',
                        name='Train', line=dict(color='#374151', dash='dot', width=2),
                        marker=dict(size=7)
                    ))
                fig.add_trace(go.Scatter(
                    x=fold_nums, y=fold_vals, mode='lines+markers',
                    name='CV', line=dict(color=color, width=3),
                    marker=dict(size=10, symbol='diamond')
                ))
                fig.add_hline(y=fold_vals.mean(), line_dash='dash', line_color=color,
                               annotation_text=f"Mean={fold_vals.mean()*100:.2f}%")
                fig.update_layout(
                    title=f"{label} per Fold",
                    yaxis_range=[max(0, fold_vals.min()-0.1), 1.05],
                    yaxis_tickformat='.1%', height=280, showlegend=True
                )
                dark_fig(fig)
                col.plotly_chart(fig, use_container_width=True)

            # Bias-Variance summary
            for metric, label in [('auc_cv', 'ROC-AUC'), ('acc_cv', 'Accuracy'), ('f1_cv', 'F1')]:
                cv_mean  = res[metric].mean()
                cv_std   = res[metric].std()
                tr_mean  = res[f'train_{metric}'].mean() if f'train_{metric}' in res else cv_mean
                gap      = tr_mean - cv_mean
                verdict  = "✅ Good fit" if gap < 0.05 else ("⚠️ Slight overfit" if gap < 0.12 else "❌ Overfit")
                st.markdown(
                    f"<div class='info' style='padding:8px 14px'>"
                    f"<b>{label}:</b> Train={tr_mean*100:.1f}%  CV={cv_mean*100:.1f}%  "
                    f"Gap={gap*100:.1f}%  Std={cv_std*100:.2f}%  →  {verdict}</div>",
                    unsafe_allow_html=True
                )

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ROC & PR CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 ROC & PR Curves":
    st.markdown("## 📈 ROC & Precision-Recall Curves")

    st.markdown('<div class="info">ℹ️ Curves shown on full training set (all 891 samples). '
                'Use CV AUC values for unbiased estimates.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">ROC Curves — All Models</div>',
                    unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                          line=dict(color='#374151', dash='dash', width=1.5))

        for name in sorted(probas.keys(), key=lambda k: results[k]['auc_cv'].mean(), reverse=True):
            fpr, tpr, _ = roc_curve(y_arr, probas[name])
            auc = roc_auc_score(y_arr, probas[name])
            is_best = name == best_name
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc:.3f})",
                line=dict(color=MODEL_COLORS[name], width=3.5 if is_best else 1.8,
                          dash='solid' if is_best else 'solid'),
                opacity=1.0 if is_best else 0.75
            ))

        fig_roc.update_layout(
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            title='ROC Curves — Train Set AUC',
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
            legend=dict(x=0.62, y=0.05, bgcolor='rgba(17,24,39,0.8)',
                        bordercolor='#374151', borderwidth=1)
        )
        dark_fig(fig_roc, 520)
        st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Precision-Recall Curves</div>',
                    unsafe_allow_html=True)
        fig_pr = go.Figure()
        baseline = y_arr.mean()
        fig_pr.add_hline(y=baseline, line_dash='dash', line_color='#374151',
                         annotation_text=f'Baseline={baseline:.2f}')

        for name in sorted(probas.keys(), key=lambda k: results[k]['auc_cv'].mean(), reverse=True):
            prec, rec, _ = precision_recall_curve(y_arr, probas[name])
            is_best = name == best_name
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec, mode='lines', name=name,
                line=dict(color=MODEL_COLORS[name], width=3.5 if is_best else 1.8),
                opacity=1.0 if is_best else 0.75
            ))

        fig_pr.update_layout(
            xaxis_title='Recall', yaxis_title='Precision',
            title='Precision-Recall Curves',
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
            legend=dict(x=0.01, y=0.05, bgcolor='rgba(17,24,39,0.8)',
                        bordercolor='#374151', borderwidth=1)
        )
        dark_fig(fig_pr, 520)
        st.plotly_chart(fig_pr, use_container_width=True)

    # Threshold Analysis for best model
    st.markdown(f'<div class="section-header">Threshold Analysis — {best_name}</div>',
                unsafe_allow_html=True)
    thresholds = np.linspace(0.1, 0.9, 81)
    prec_arr, rec_arr, f1_arr, acc_arr = [], [], [], []
    for t in thresholds:
        preds = (probas[best_name] >= t).astype(int)
        prec_arr.append(prec if (prec := (preds & y_arr).sum() / (preds.sum() + 1e-9)) else 0)
        rec_arr.append((preds & y_arr).sum() / (y_arr.sum() + 1e-9))
        f1_arr.append(f1_score(y_arr, preds, zero_division=0))
        acc_arr.append(accuracy_score(y_arr, preds))

    fig_th = go.Figure()
    for vals, name_m, color in [(prec_arr, 'Precision', C_BLUE),
                                  (rec_arr,  'Recall',    C_GREEN),
                                  (f1_arr,   'F1-Score',  C_YELLOW),
                                  (acc_arr,  'Accuracy',  C_PINK)]:
        fig_th.add_trace(go.Scatter(x=thresholds, y=vals, mode='lines',
                                     name=name_m, line=dict(color=color, width=2.5)))
    fig_th.add_vline(x=0.5, line_dash='dash', line_color='white',
                      annotation_text='Default=0.5')
    best_f1_idx = np.argmax(f1_arr)
    fig_th.add_vline(x=thresholds[best_f1_idx], line_dash='dot', line_color=C_YELLOW,
                      annotation_text=f'Best F1 @ {thresholds[best_f1_idx]:.2f}')
    fig_th.update_layout(xaxis_title='Decision Threshold',
                          yaxis_title='Metric Value', yaxis_range=[0, 1.05],
                          title=f'Threshold vs Metrics — {best_name}')
    dark_fig(fig_th, 400)
    st.plotly_chart(fig_th, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.markdown("## 🔍 Feature Importance Analysis")

    feat_labels = [FEATURE_LABELS.get(f, f) for f in FEATURES]

    tabs = st.tabs(list(feat_imp.keys()) + ["🔀 Combined Ranking"])

    for tab, (model_name, importances) in zip(tabs[:-1], feat_imp.items()):
        with tab:
            imp_df = pd.DataFrame({
                'Feature':    feat_labels,
                'Importance': importances,
                'Feature_raw': FEATURES
            }).sort_values('Importance', ascending=True)

            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                         color='Importance',
                         color_continuous_scale='Viridis',
                         text=imp_df['Importance'].apply(lambda x: f"{x:.4f}"),
                         title=f"{model_name} — Feature Importance")
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False, yaxis_title='',
                               xaxis_title='Importance Score')
            dark_fig(fig, 500)
            tab.plotly_chart(fig, use_container_width=True)

            # Top insight
            top3 = imp_df.sort_values('Importance', ascending=False).head(3)
            insight_txt = " → ".join(
                [f"<b>{r['Feature']}</b> ({r['Importance']:.4f})"
                 for _, r in top3.iterrows()]
            )
            tab.markdown(f'<div class="insight">🔝 Top 3: {insight_txt}</div>',
                         unsafe_allow_html=True)

    # Combined ranking tab
    with tabs[-1]:
        st.markdown('<div class="section-header">Normalized Importance — All Models</div>',
                    unsafe_allow_html=True)
        imp_matrix = pd.DataFrame(index=feat_labels)
        for model_name, importances in feat_imp.items():
            norm = np.array(importances)
            norm = norm / (norm.sum() + 1e-9)
            imp_matrix[model_name] = norm

        imp_matrix['Average'] = imp_matrix.mean(axis=1)
        imp_matrix = imp_matrix.sort_values('Average', ascending=False)

        fig_heat = px.imshow(
            imp_matrix.drop('Average', axis=1).T,
            color_continuous_scale='YlOrRd',
            aspect='auto', title='Feature Importance Heatmap (all models)',
            text_auto='.3f'
        )
        fig_heat.update_traces(textfont_size=8)
        dark_fig(fig_heat, 380)
        st.plotly_chart(fig_heat, use_container_width=True)

        fig_avg = px.bar(
            imp_matrix.reset_index().rename(columns={'index': 'Feature'}),
            x='Feature', y='Average',
            color='Average', color_continuous_scale='Viridis',
            text=imp_matrix['Average'].apply(lambda x: f"{x:.4f}").values,
            title='Average Normalized Importance Across All Models'
        )
        fig_avg.update_traces(textposition='outside')
        fig_avg.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
        dark_fig(fig_avg, 400)
        st.plotly_chart(fig_avg, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LEARNING CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Learning Curves":
    st.markdown("## 📉 Learning Curves — Bias vs Variance")

    sel_model = st.selectbox("Select Model", list(trained.keys()),
                              index=list(trained.keys()).index(best_name))
    model_lc  = get_models()[sel_model]
    color     = MODEL_COLORS[sel_model]

    with st.spinner(f"Computing learning curve for {sel_model}..."):
        train_sizes, train_scores, val_scores = learning_curve(
            model_lc, X_sc, y_arr,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            train_sizes=np.linspace(0.1, 1.0, 12),
            scoring='roc_auc', n_jobs=-1
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_scores.mean(axis=1),
        mode='lines+markers', name='Train AUC',
        line=dict(color='#374151', width=2.5, dash='dot'),
        error_y=dict(type='data', array=train_scores.std(axis=1), visible=True,
                     color='#374151')
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_scores.mean(axis=1),
        mode='lines+markers', name='CV AUC',
        line=dict(color=color, width=3),
        marker=dict(size=9, symbol='diamond'),
        error_y=dict(type='data', array=val_scores.std(axis=1), visible=True,
                     color=color)
    ))
    # Shaded bands
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_scores.mean(axis=1) + train_scores.std(axis=1),
                          (train_scores.mean(axis=1) - train_scores.std(axis=1))[::-1]]),
        fill='toself', fillcolor='rgba(55,65,81,0.2)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_scores.mean(axis=1) + val_scores.std(axis=1),
                          (val_scores.mean(axis=1) - val_scores.std(axis=1))[::-1]]),
        fill='toself', fillcolor=f'rgba(99,102,241,0.12)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False
    ))

    final_gap = train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]
    verdict = "✅ Good fit" if final_gap < 0.05 else ("⚠️ Slight overfitting" if final_gap < 0.12 else "❌ Overfitting")
    fig.update_layout(xaxis_title='Training Set Size', yaxis_title='ROC-AUC',
                      title=f'Learning Curve — {sel_model}  |  Gap={final_gap*100:.1f}%  {verdict}')
    dark_fig(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Train AUC", f"{train_scores.mean(axis=1)[-1]*100:.2f}%")
    col2.metric("Final CV AUC",    f"{val_scores.mean(axis=1)[-1]*100:.2f}%")
    col3.metric("Train-CV Gap",    f"{final_gap*100:.2f}%",
                delta_color="inverse")

    if final_gap > 0.12:
        st.markdown('<div class="warning">⚠️ High gap suggests overfitting. '
                    'Try reducing max_depth, increasing min_samples, or adding regularization.</div>',
                    unsafe_allow_html=True)
    elif final_gap < 0.02:
        st.markdown('<div class="warning">⚠️ Very low gap might suggest underfitting. '
                    'Try increasing model complexity or adding more features.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight">✅ Healthy bias-variance tradeoff — '
                    'model generalizes well.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧩 Confusion Matrices":
    st.markdown("## 🧩 Confusion Matrices & Classification Reports")

    model_names = list(trained.keys())
    n_cols = 3
    rows = [model_names[i:i+n_cols] for i in range(0, len(model_names), n_cols)]

    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)

    for row_models in rows:
        cols = st.columns(n_cols)
        for col, name in zip(cols, row_models):
            preds  = (probas[name] >= threshold).astype(int)
            cm     = confusion_matrix(y_arr, preds)
            tn, fp, fn, tp = cm.ravel()
            acc    = accuracy_score(y_arr, preds)
            auc    = roc_auc_score(y_arr, probas[name])
            f1     = f1_score(y_arr, preds, zero_division=0)
            color  = MODEL_COLORS[name]

            fig = px.imshow(
                cm, text_auto=True,
                color_continuous_scale=[[0, '#111827'], [0.5, '#1f2937'], [1, color]],
                x=['Pred: Not Survived', 'Pred: Survived'],
                y=['True: Not Survived', 'True: Survived'],
                title=f"{name}<br><sup>AUC={auc:.3f} | Acc={acc:.3f} | F1={f1:.3f}</sup>",
                aspect='equal'
            )
            fig.update_traces(texttemplate='<b>%{z}</b>', textfont_size=18)
            fig.update_layout(coloraxis_showscale=False, height=310, margin=dict(t=70, b=10))
            dark_fig(fig)
            col.plotly_chart(fig, use_container_width=True)

            # Per-class metrics
            prec_0 = tn / (tn + fn + 1e-9)
            rec_0  = tn / (tn + fp + 1e-9)
            prec_1 = tp / (tp + fp + 1e-9)
            rec_1  = tp / (tp + fn + 1e-9)
            col.markdown(f"""
            <div style='font-size:0.75em;color:#9ca3af;padding:0 4px'>
              Not Survived → Precision:{prec_0:.2f} Recall:{rec_0:.2f}<br>
              Survived &nbsp;&nbsp;&nbsp;→ Precision:{prec_1:.2f} Recall:{rec_1:.2f}
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.markdown("## 🔮 Live Passenger Survival Prediction")

    c_left, c_right = st.columns([1, 1.4])

    with c_left:
        st.markdown('<div class="section-header">Passenger Details</div>',
                    unsafe_allow_html=True)

        pclass   = st.selectbox("Passenger Class", [1, 2, 3],
                                format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class")
        sex      = st.radio("Gender", ["male", "female"], horizontal=True)
        age      = st.slider("Age", 0, 80, 30)
        fare     = st.slider("Fare (£)", 0, 520, 50)
        sibsp    = st.slider("Siblings / Spouses aboard", 0, 8, 0)
        parch    = st.slider("Parents / Children aboard", 0, 6, 0)
        embarked = st.selectbox("Port of Embarkation",
                                ["S — Southampton", "C — Cherbourg", "Q — Queenstown"])
        title    = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
        has_cabin = st.checkbox("Has Cabin Assignment?", False)

        sel_model_pred = st.selectbox("Model to Use", list(trained.keys()),
                                       index=list(trained.keys()).index(best_name))
        threshold_pred = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)
        predict_btn    = st.button("🔮 Predict Survival", use_container_width=True, type="primary")

    with c_right:
        if predict_btn:
            # Build feature vector
            sex_enc     = 1 if sex == "female" else 0
            emb_enc     = {"S — Southampton": 0, "C — Cherbourg": 1, "Q — Queenstown": 2}[embarked]
            title_enc   = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}[title]
            family_size = sibsp + parch + 1
            is_alone    = 1 if family_size == 1 else 0
            family_cat  = 0 if family_size == 1 else (1 if family_size <= 4 else 2)
            log_fare    = np.log1p(fare)
            fare_pp     = fare / family_size
            cabin_flag  = 1 if has_cabin else 0

            passenger = np.array([[
                pclass, sex_enc, float(age), sibsp, parch,
                float(fare), log_fare, family_size, is_alone,
                cabin_flag, emb_enc, title_enc, family_cat, fare_pp
            ]], dtype=float)

            passenger_sc = scaler.transform(passenger)

            # All models
            all_preds = {}
            for m_name, m in trained.items():
                prob = m.predict_proba(passenger_sc)[0, 1]
                pred = int(prob >= threshold_pred)
                all_preds[m_name] = {'prob': prob, 'pred': pred}

            # Selected model result
            sel_prob = all_preds[sel_model_pred]['prob']
            sel_pred = all_preds[sel_model_pred]['pred']

            # Main prediction card
            if sel_pred == 1:
                st.markdown(f"""
                <div class="predict-survived">
                  <div style="font-size:3.5em">🟢</div>
                  <div style="font-size:1.8em;font-weight:800;color:{C_GREEN};margin:10px 0">
                    SURVIVED
                  </div>
                  <div style="font-size:1.1em;color:#a7f3d0">
                    Survival Probability: <b>{sel_prob*100:.1f}%</b>
                  </div>
                  <div style="font-size:0.85em;color:#6ee7b7;margin-top:8px">
                    Model: {sel_model_pred} &nbsp;|&nbsp; Threshold: {threshold_pred:.2f}
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="predict-not">
                  <div style="font-size:3.5em">🔴</div>
                  <div style="font-size:1.8em;font-weight:800;color:{C_RED};margin:10px 0">
                    DID NOT SURVIVE
                  </div>
                  <div style="font-size:1.1em;color:#fca5a5">
                    Survival Probability: <b>{sel_prob*100:.1f}%</b>
                  </div>
                  <div style="font-size:0.85em;color:#f87171;margin-top:8px">
                    Model: {sel_model_pred} &nbsp;|&nbsp; Threshold: {threshold_pred:.2f}
                  </div>
                </div>""", unsafe_allow_html=True)

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sel_prob * 100,
                delta={'reference': 38.4, 'increasing': {'color': C_GREEN},
                       'decreasing': {'color': C_RED}},
                title={'text': "Survival Probability %", 'font': {'color': '#f9fafb'}},
                number={'font': {'color': C_GREEN if sel_pred == 1 else C_RED, 'size': 48}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#6b7280'},
                    'bar':  {'color': C_GREEN if sel_pred == 1 else C_RED},
                    'bgcolor': '#1f2937',
                    'bordercolor': '#374151',
                    'steps': [
                        {'range': [0, 38.4], 'color': 'rgba(239,68,68,0.15)'},
                        {'range': [38.4, 100], 'color': 'rgba(16,185,129,0.15)'},
                    ],
                    'threshold': {'value': 38.4, 'line': {'color': '#6b7280', 'width': 2},
                                  'thickness': 0.8}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='#111827', font_color='#f9fafb', height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # All models comparison
            st.markdown('<div class="section-header">All Models — Probability Comparison</div>',
                        unsafe_allow_html=True)
            all_df = pd.DataFrame([
                {'Model': n, 'Probability': d['prob'], 'Verdict': 'Survived' if d['pred'] else 'Not Survived'}
                for n, d in sorted(all_preds.items(), key=lambda x: x[1]['prob'], reverse=True)
            ])
            fig_all = px.bar(all_df, x='Model', y='Probability',
                             color='Verdict', text=all_df['Probability'].apply(lambda x: f"{x*100:.1f}%"),
                             color_discrete_map={'Survived': C_GREEN, 'Not Survived': C_RED},
                             range_y=[0, 1])
            fig_all.add_hline(y=threshold_pred, line_dash='dash', line_color=C_YELLOW,
                               annotation_text=f'Threshold={threshold_pred:.2f}')
            fig_all.update_traces(textposition='outside')
            fig_all.update_layout(yaxis_tickformat='.0%', showlegend=True,
                                   xaxis_tickangle=-25)
            dark_fig(fig_all, 350)
            st.plotly_chart(fig_all, use_container_width=True)

            # Ensemble vote
            votes_survived = sum(1 for d in all_preds.values() if d['pred'] == 1)
            votes_not      = len(all_preds) - votes_survived
            ensemble_verdict = "SURVIVED" if votes_survived > votes_not else "NOT SURVIVED"
            avg_prob = np.mean([d['prob'] for d in all_preds.values()])
            st.markdown(f"""
            <div class="{'insight' if votes_survived > votes_not else 'warning'}">
              <b>📊 Ensemble Verdict (majority vote):</b> {ensemble_verdict}<br>
              Models voting Survived: {votes_survived}/{len(all_preds)} &nbsp;|&nbsp;
              Average Probability: {avg_prob*100:.1f}%
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="info" style="text-align:center;padding:40px">
              <div style="font-size:3em">🔮</div>
              <div style="font-size:1.1em;margin-top:14px">
                Fill in passenger details on the left<br>and click <b>Predict Survival</b>
              </div>
              <div style="font-size:0.82em;color:#6b7280;margin-top:10px">
                7 models will all give their prediction simultaneously
              </div>
            </div>""", unsafe_allow_html=True)

            # Historical reference
            st.markdown('<div class="section-header">Historical Reference Rates</div>',
                        unsafe_allow_html=True)
            ref_data = {
                'Group':   ['1st Class F', '1st Class M', '2nd Class F', '2nd Class M',
                            '3rd Class F', '3rd Class M', 'Children', 'Overall'],
                'Rate %':  [97, 33, 86, 14, 46, 16, 58, 38],
            }
            ref_df = pd.DataFrame(ref_data).sort_values('Rate %', ascending=True)
            fig_ref = px.bar(ref_df, x='Rate %', y='Group', orientation='h',
                              color='Rate %', color_continuous_scale='RdYlGn',
                              text='Rate %', title='Historical Survival Rates')
            fig_ref.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_ref.update_layout(coloraxis_showscale=False)
            dark_fig(fig_ref, 380)
            st.plotly_chart(fig_ref, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EXPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💾 Export Models":
    st.markdown("## 💾 Export Trained Models")

    st.markdown('<div class="info">💡 Models are trained on the full dataset (891 samples) '
                'with StandardScaler. Always use the scaler before calling predict().</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    export_data = {
        'models':   trained,
        'scaler':   scaler,
        'features': FEATURES,
        'feature_labels': FEATURE_LABELS,
        'best_model': best_name,
        'cv_results': {
            name: {
                'auc_mean': float(res['auc_cv'].mean()),
                'auc_std':  float(res['auc_cv'].std()),
                'acc_mean': float(res['acc_cv'].mean()),
                'f1_mean':  float(res['f1_cv'].mean()),
            }
            for name, res in results.items()
        }
    }

    with c1:
        st.markdown('<div class="section-header">Export All Models (Bundle)</div>',
                    unsafe_allow_html=True)
        buf = io.BytesIO()
        pickle.dump(export_data, buf)
        st.download_button(
            "⬇️ Download All Models (.pkl)",
            data=buf.getvalue(),
            file_name="titanic_ml_bundle.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )

        st.markdown('<div class="section-header">Export Individual Model</div>',
                    unsafe_allow_html=True)
        sel_export = st.selectbox("Select Model", list(trained.keys()))
        buf2 = io.BytesIO()
        pickle.dump({'model': trained[sel_export], 'scaler': scaler,
                     'features': FEATURES}, buf2)
        st.download_button(
            f"⬇️ Download {sel_export} (.pkl)",
            data=buf2.getvalue(),
            file_name=f"titanic_{sel_export.lower().replace(' ','_')}.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )

    with c2:
        st.markdown('<div class="section-header">CV Results — JSON Export</div>',
                    unsafe_allow_html=True)
        import json
        json_results = json.dumps(export_data['cv_results'], indent=2)
        st.download_button(
            "⬇️ Download CV Results (.json)",
            data=json_results.encode(),
            file_name="titanic_cv_results.json",
            mime="application/json",
            use_container_width=True
        )

        st.markdown('<div class="section-header">Processed Dataset (CSV)</div>',
                    unsafe_allow_html=True)
        export_df = X.copy()
        export_df.columns = [FEATURE_LABELS.get(c, c) for c in export_df.columns]
        export_df['Survived'] = y_arr
        csv_data = export_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Processed Dataset (.csv)",
            data=csv_data,
            file_name="titanic_processed.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Usage code
        st.markdown('<div class="section-header">🐍 Usage Code</div>', unsafe_allow_html=True)
        st.code(f"""
import pickle
import numpy as np

# Load bundle
with open('titanic_ml_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

scaler   = bundle['scaler']
model    = bundle['models']['{best_name}']
features = bundle['features']

# Prepare a passenger (same feature order)
# features = {FEATURES}
passenger = np.array([[3, 1, 25, 0, 0, 10.0,
                        2.30, 1, 1, 0, 0, 1, 0, 10.0]])
X_sc = scaler.transform(passenger)
prob = model.predict_proba(X_sc)[0, 1]
print(f"Survival probability: {{prob:.2%}}")
""", language='python')

    # Summary table
    st.markdown('<div class="section-header">Model Performance Summary</div>',
                unsafe_allow_html=True)
    sum_df = pd.DataFrame([
        {
            'Model':    name,
            'ROC-AUC':  f"{res['auc_cv'].mean()*100:.2f}% ± {res['auc_cv'].std()*100:.2f}",
            'Accuracy': f"{res['acc_cv'].mean()*100:.2f}% ± {res['acc_cv'].std()*100:.2f}",
            'F1-Score': f"{res['f1_cv'].mean()*100:.2f}% ± {res['f1_cv'].std()*100:.2f}",
            'Best':     '🏆' if name == best_name else '',
        }
        for name, res in sorted(results.items(), key=lambda kv: kv[1]['auc_cv'].mean(), reverse=True)
    ])
    st.dataframe(sum_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    st.markdown("## 🧠 Model Explainability — Feature Contributions & Decision Analysis")

    st.markdown("""
    <div class="info">
      <b>🔬 SHAP-equivalent Analysis</b> (without the SHAP library)<br>
      Local contribution = how much each feature <b>pushes the prediction up or down</b>
      from the baseline for a specific passenger.<br>
      Formula: <code>contribution[i] = P(x) − P(x with feature_i replaced by population mean)</code>
    </div>
    """, unsafe_allow_html=True)

    # ── Helper ───────────────────────────────────────────────────────────────
    feat_labels_list = [FEATURE_LABELS.get(f, f) for f in FEATURES]

    @st.cache_data(show_spinner=False)
    def compute_contributions(_model, _X_sc, sample_idx):
        sample      = _X_sc[sample_idx:sample_idx+1].copy()
        sample_pred = _model.predict_proba(sample)[0, 1]
        baseline    = _model.predict_proba(_X_sc).mean(axis=0)[1]
        contribs    = []
        for i in range(_X_sc.shape[1]):
            x_mod       = sample.copy()
            x_mod[0, i] = _X_sc[:, i].mean()
            contribs.append(sample_pred - _model.predict_proba(x_mod)[0, 1])
        return np.array(contribs), sample_pred, baseline

    @st.cache_data(show_spinner=False)
    def compute_population_contributions(_model, _X_sc, n_sample=200):
        rng     = np.random.RandomState(42)
        idxs    = rng.choice(len(_X_sc), min(n_sample, len(_X_sc)), replace=False)
        baseline = _model.predict_proba(_X_sc).mean(axis=0)[1]
        matrix  = np.zeros((len(idxs), _X_sc.shape[1]))
        for k, idx in enumerate(idxs):
            sample      = _X_sc[idx:idx+1].copy()
            sample_pred = _model.predict_proba(sample)[0, 1]
            for i in range(_X_sc.shape[1]):
                x_mod       = sample.copy()
                x_mod[0, i] = _X_sc[:, i].mean()
                matrix[k, i] = sample_pred - _model.predict_proba(x_mod)[0, 1]
        return matrix, baseline

    @st.cache_data(show_spinner=False)
    def compute_permutation_imp(_X_sc, _y):
        from sklearn.inspection import permutation_importance as pi
        result = {}
        for name in ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'SVM']:
            r = pi(trained[name], _X_sc, _y, n_repeats=15,
                   scoring='roc_auc', random_state=42, n_jobs=-1)
            result[name] = (r.importances_mean, r.importances_std)
        return result

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Individual Explanation",
        "👥 Population Analysis",
        "🔀 Permutation Importance",
        "🌳 Decision Tree Rules",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — Individual Explanation
    # ════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">Choose a Passenger to Explain</div>',
                    unsafe_allow_html=True)

        c_ctrl, c_info = st.columns([1, 1.5])
        with c_ctrl:
            expl_model_name = st.selectbox(
                "Model", list(trained.keys()),
                index=list(trained.keys()).index(best_name),
                key="expl_model"
            )

            mode = st.radio("Select passenger by:", ["Index", "Profile"], horizontal=True)

            if mode == "Index":
                passenger_idx = st.slider("Passenger index", 0, len(X)-1, 0)
            else:
                pclass_f = st.selectbox("Class", [1, 2, 3])
                sex_f    = st.radio("Sex", ["female", "male"], horizontal=True)
                survived_f = st.radio("Actual outcome", ["Survived", "Not Survived"],
                                      horizontal=True)
                survived_val = 1 if survived_f == "Survived" else 0
                mask = ((eng_df['Pclass'] == pclass_f) &
                        (eng_df['Sex'] == sex_f) &
                        (eng_df['Survived'] == survived_val))
                candidates = eng_df[mask]
                if len(candidates) == 0:
                    st.warning("No passengers match this profile.")
                    passenger_idx = 0
                else:
                    passenger_idx = int(candidates.index[0])

            explain_btn = st.button("🔬 Explain This Passenger", use_container_width=True,
                                     type="primary")

        with c_info:
            row = eng_df.iloc[passenger_idx]
            actual = "✅ SURVIVED" if row['Survived'] == 1 else "❌ NOT SURVIVED"
            color_actual = C_GREEN if row['Survived'] == 1 else C_RED
            st.markdown(f"""
            <div class="model-card">
              <div style="font-size:0.8em;color:#6b7280;margin-bottom:10px">PASSENGER INFO</div>
              <div style="font-size:1.05em;font-weight:700;color:#f9fafb;margin-bottom:8px">
                {str(row['Name'])[:55]}
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.84em">
                <div><span style="color:#6b7280">Class:</span> <b>{row['Pclass']}</b></div>
                <div><span style="color:#6b7280">Sex:</span> <b>{row['Sex']}</b></div>
                <div><span style="color:#6b7280">Age:</span> <b>{row['Age']}</b></div>
                <div><span style="color:#6b7280">Fare:</span> <b>£{row['Fare']:.1f}</b></div>
                <div><span style="color:#6b7280">Embarked:</span> <b>{row['Embarked']}</b></div>
                <div><span style="color:#6b7280">Family:</span> <b>{int(row['Family_Size'])}</b></div>
                <div><span style="color:#6b7280">Title:</span> <b>{row['Title']}</b></div>
                <div><span style="color:#6b7280">Cabin:</span> <b>{'Yes' if row['Has_Cabin'] else 'No'}</b></div>
              </div>
              <div style="margin-top:14px;font-size:1.15em;font-weight:800;color:{color_actual}">
                {actual}
              </div>
            </div>
            """, unsafe_allow_html=True)

        if explain_btn or True:  # auto-compute on load
            with st.spinner("Computing feature contributions..."):
                expl_model  = trained[expl_model_name]
                contribs, pred_prob, baseline = compute_contributions(
                    expl_model, X_sc, passenger_idx)

            # Waterfall chart
            st.markdown('<div class="section-header">Feature Contribution Waterfall</div>',
                        unsafe_allow_html=True)

            order      = np.argsort(np.abs(contribs))[::-1]
            top_n      = 10
            top_labels = [feat_labels_list[i] for i in order[:top_n]]
            top_vals   = contribs[order[:top_n]]
            top_colors = [C_GREEN if v >= 0 else C_RED for v in top_vals]

            # Running total for waterfall
            running = baseline
            waterfall_x, waterfall_base, waterfall_val, waterfall_color = [], [], [], []
            for label, val in zip(top_labels[::-1], top_vals[::-1]):
                waterfall_x.append(label)
                waterfall_base.append(running if val >= 0 else running + val)
                waterfall_val.append(abs(val))
                waterfall_color.append(C_GREEN if val >= 0 else C_RED)
                running += val

            fig_wf = go.Figure()
            # Baseline invisible bar
            fig_wf.add_trace(go.Bar(
                x=waterfall_x, y=waterfall_base, marker_color='rgba(0,0,0,0)',
                showlegend=False, hoverinfo='skip'
            ))
            # Contribution bars
            fig_wf.add_trace(go.Bar(
                x=waterfall_x, y=waterfall_val, marker_color=waterfall_color,
                name='Contribution', opacity=0.9,
                text=[f"{'+' if v>=0 else ''}{v:.3f}" for v in top_vals[::-1]],
                textposition='outside'
            ))
            fig_wf.add_hline(y=baseline, line_dash='dot', line_color=C_YELLOW,
                              annotation_text=f"Baseline={baseline*100:.1f}%")
            fig_wf.add_hline(y=pred_prob, line_dash='dash', line_color=C_GREEN if pred_prob>=0.5 else C_RED,
                              annotation_text=f"Prediction={pred_prob*100:.1f}%")
            fig_wf.update_layout(barmode='stack', yaxis_range=[0, 1.1],
                                  yaxis_tickformat='.0%',
                                  title=f"Feature Contributions — {expl_model_name}",
                                  showlegend=False, xaxis_tickangle=-25)
            dark_fig(fig_wf, 430)
            st.plotly_chart(fig_wf, use_container_width=True)

            # Horizontal sorted bar
            col_l, col_r = st.columns(2)
            with col_l:
                idx_sorted = np.argsort(contribs)
                fig_h = go.Figure(go.Bar(
                    x=contribs[idx_sorted],
                    y=[feat_labels_list[i] for i in idx_sorted],
                    orientation='h',
                    marker_color=[C_GREEN if v >= 0 else C_RED for v in contribs[idx_sorted]],
                    text=[f"{'+' if v>=0 else ''}{v:.3f}" for v in contribs[idx_sorted]],
                    textposition='outside', opacity=0.88
                ))
                fig_h.add_vline(x=0, line_color='white', line_width=1.5)
                fig_h.update_layout(
                    title="All Feature Contributions",
                    xaxis_title="Probability change",
                    yaxis_title=""
                )
                dark_fig(fig_h, 420)
                st.plotly_chart(fig_h, use_container_width=True)

            with col_r:
                # Gauge
                verdict   = "SURVIVED" if pred_prob >= 0.5 else "NOT SURVIVED"
                v_color   = C_GREEN if pred_prob >= 0.5 else C_RED
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_prob * 100,
                    title={'text': f"Survival Probability<br><span style='font-size:0.8em;color:{v_color}'>{verdict}</span>"},
                    number={'font': {'color': v_color, 'size': 52}, 'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar':  {'color': v_color},
                        'bgcolor': '#1f2937',
                        'bordercolor': '#374151',
                        'steps': [
                            {'range': [0, 50],   'color': 'rgba(239,68,68,0.15)'},
                            {'range': [50, 100], 'color': 'rgba(16,185,129,0.15)'},
                        ],
                        'threshold': {
                            'value': baseline * 100,
                            'line': {'color': C_YELLOW, 'width': 2},
                            'thickness': 0.8
                        }
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='#111827', font_color='#f9fafb', height=280)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Verdict cards
                st.markdown(f"""
                <div class="{'insight' if pred_prob >= 0.5 else 'warning'}">
                  <b>Baseline:</b> {baseline*100:.1f}% (avg passenger)<br>
                  <b>Prediction:</b> {pred_prob*100:.1f}%<br>
                  <b>Net push:</b> {(pred_prob-baseline)*100:+.1f}%<br>
                  <b>Actual:</b> {'✅ SURVIVED' if eng_df.iloc[passenger_idx]['Survived']==1 else '❌ NOT SURVIVED'}
                </div>
                """, unsafe_allow_html=True)

                # Top 3 explanation in plain language
                top3_pos = [(feat_labels_list[i], contribs[i]) for i in np.argsort(contribs)[-3:][::-1] if contribs[i] > 0.005]
                top3_neg = [(feat_labels_list[i], contribs[i]) for i in np.argsort(contribs)[:3] if contribs[i] < -0.005]
                if top3_pos:
                    pos_text = ", ".join([f"<b>{n}</b> (+{v:.2f})" for n, v in top3_pos])
                    st.markdown(f'<div class="insight">📈 Helped survival: {pos_text}</div>',
                                unsafe_allow_html=True)
                if top3_neg:
                    neg_text = ", ".join([f"<b>{n}</b> ({v:.2f})" for n, v in top3_neg])
                    st.markdown(f'<div class="warning">📉 Hurt survival: {neg_text}</div>',
                                unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — Population Analysis
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">Population-Level Feature Contributions (200 passengers)</div>',
                    unsafe_allow_html=True)

        pop_model_name = st.selectbox("Model", list(trained.keys()),
                                       index=list(trained.keys()).index(best_name),
                                       key="pop_model")

        with st.spinner("Computing contributions for 200 passengers..."):
            contrib_matrix, pop_baseline = compute_population_contributions(
                trained[pop_model_name], X_sc, n_sample=200)

        mean_abs   = np.abs(contrib_matrix).mean(axis=0)
        mean_pos   = np.where(contrib_matrix > 0, contrib_matrix, 0).mean(axis=0)
        mean_neg   = np.where(contrib_matrix < 0, contrib_matrix, 0).mean(axis=0)
        idx_sorted = np.argsort(mean_abs)

        c1, c2 = st.columns(2)
        with c1:
            # Mean absolute contribution
            fig_ma = go.Figure(go.Bar(
                x=mean_abs[idx_sorted],
                y=[feat_labels_list[i] for i in idx_sorted],
                orientation='h',
                marker=dict(
                    color=mean_abs[idx_sorted],
                    colorscale='Viridis', showscale=True,
                    colorbar=dict(title='|Contribution|')
                ),
                text=[f"{v:.4f}" for v in mean_abs[idx_sorted]],
                textposition='outside', opacity=0.9
            ))
            fig_ma.update_layout(title=f"Mean |Contribution| — {pop_model_name}<br>"
                                       f"<sup>Global feature importance via contribution analysis</sup>",
                                  xaxis_title="Mean absolute probability change")
            dark_fig(fig_ma, 480)
            st.plotly_chart(fig_ma, use_container_width=True)

        with c2:
            # Positive vs Negative breakdown
            fig_pn = go.Figure()
            fig_pn.add_trace(go.Bar(
                x=[feat_labels_list[i] for i in idx_sorted],
                y=mean_pos[idx_sorted],
                name='Avg positive push', marker_color=C_GREEN, opacity=0.85
            ))
            fig_pn.add_trace(go.Bar(
                x=[feat_labels_list[i] for i in idx_sorted],
                y=mean_neg[idx_sorted],
                name='Avg negative push', marker_color=C_RED, opacity=0.85
            ))
            fig_pn.add_hline(y=0, line_color='white', line_width=1)
            fig_pn.update_layout(barmode='relative',
                                   title="Direction of Contribution (Population Average)",
                                   yaxis_title="Avg probability change",
                                   xaxis_tickangle=-35)
            dark_fig(fig_pn, 480)
            st.plotly_chart(fig_pn, use_container_width=True)

        # Contribution distribution — box plots per feature
        st.markdown('<div class="section-header">Contribution Distribution per Feature</div>',
                    unsafe_allow_html=True)

        top8_idx = np.argsort(mean_abs)[-8:][::-1]
        fig_box  = go.Figure()
        for i in top8_idx:
            fig_box.add_trace(go.Box(
                y=contrib_matrix[:, i],
                name=feat_labels_list[i],
                marker_color=MODEL_COLORS[pop_model_name],
                boxmean='sd', opacity=0.85
            ))
        fig_box.add_hline(y=0, line_color='white', line_dash='dash', line_width=1)
        fig_box.update_layout(
            title="Feature Contribution Distribution — Top 8 Features<br>"
                  "<sup>Each box shows the spread of contributions across 200 passengers</sup>",
            yaxis_title="Contribution (probability change)",
            showlegend=False, xaxis_tickangle=-20
        )
        dark_fig(fig_box, 420)
        st.plotly_chart(fig_box, use_container_width=True)

        # Survival group comparison
        st.markdown('<div class="section-header">Feature Contributions: Survivors vs Non-Survivors</div>',
                    unsafe_allow_html=True)

        rng      = np.random.RandomState(42)
        pop_idxs = rng.choice(len(X_sc), min(200, len(X_sc)), replace=False)
        pop_y    = y_arr[pop_idxs]
        survived_mask = pop_y == 1

        mean_surv  = contrib_matrix[survived_mask].mean(axis=0)
        mean_dead  = contrib_matrix[~survived_mask].mean(axis=0)
        idx_s      = np.argsort(np.abs(mean_surv - mean_dead))[::-1][:10]

        fig_grp = go.Figure()
        fig_grp.add_trace(go.Bar(
            name='Survivors',     x=[feat_labels_list[i] for i in idx_s],
            y=mean_surv[idx_s],   marker_color=C_GREEN, opacity=0.85
        ))
        fig_grp.add_trace(go.Bar(
            name='Non-Survivors', x=[feat_labels_list[i] for i in idx_s],
            y=mean_dead[idx_s],   marker_color=C_RED, opacity=0.85
        ))
        fig_grp.update_layout(
            barmode='group',
            title="Top 10 Features — Avg Contribution Difference Between Groups",
            yaxis_title="Avg contribution", xaxis_tickangle=-25
        )
        dark_fig(fig_grp, 420)
        st.plotly_chart(fig_grp, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 3 — Permutation Importance
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">Permutation Importance — Model-Agnostic</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info">
          <b>How it works:</b> For each feature, randomly shuffle its values across all passengers.
          Measure how much the ROC-AUC drops. A large drop = feature is important.<br>
          <b>Advantage over built-in importance:</b> Works for ANY model, including SVM and Logistic Regression.
          Also accounts for feature interactions.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Computing permutation importance (15 repeats × 4 models)..."):
            perm_data = compute_permutation_imp(X_sc, y_arr)

        fig_perm = make_subplots(rows=2, cols=2,
                                  subplot_titles=list(perm_data.keys()),
                                  vertical_spacing=0.14, horizontal_spacing=0.1)

        for idx_p, (name, (means, stds)) in enumerate(perm_data.items()):
            row_p = idx_p // 2 + 1
            col_p = idx_p %  2 + 1
            order_p = np.argsort(means)
            bar_colors = [C_RED if m < 0 else MODEL_COLORS[name] for m in means[order_p]]

            fig_perm.add_trace(go.Bar(
                x=means[order_p],
                y=[feat_labels_list[i] for i in order_p],
                orientation='h',
                error_x=dict(type='data', array=stds[order_p], visible=True,
                              color='rgba(255,255,255,0.5)', thickness=1.5),
                marker_color=bar_colors, opacity=0.88,
                name=name, showlegend=False
            ), row=row_p, col=col_p)

        fig_perm.update_layout(
            title_text='Permutation Importance (AUC drop when feature shuffled)',
            height=700, paper_bgcolor='#111827', plot_bgcolor='#111827',
            font=dict(color='#f9fafb', size=10)
        )
        for i in range(1, 5):
            fig_perm.update_xaxes(gridcolor='#1f2937', row=(i-1)//2+1, col=(i-1)%2+1)
            fig_perm.update_yaxes(gridcolor='#1f2937', row=(i-1)//2+1, col=(i-1)%2+1)

        st.plotly_chart(fig_perm, use_container_width=True)

        # Aggregated comparison
        st.markdown('<div class="section-header">Permutation Importance — All Models Compared</div>',
                    unsafe_allow_html=True)
        fig_agg = go.Figure()
        for name, (means, stds) in perm_data.items():
            fig_agg.add_trace(go.Bar(
                name=name,
                x=feat_labels_list,
                y=means,
                error_y=dict(type='data', array=stds, visible=True),
                marker_color=MODEL_COLORS[name], opacity=0.82
            ))
        fig_agg.update_layout(barmode='group',
                               title='Permutation Importance Comparison Across Models',
                               yaxis_title='AUC decrease (mean ± std)',
                               xaxis_tickangle=-30)
        dark_fig(fig_agg, 440)
        st.plotly_chart(fig_agg, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4 — Decision Tree Rules
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">Decision Tree — Interpretable Rule Extraction</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info">
          A shallow Decision Tree (depth=4) trained on the same data provides
          <b>fully interpretable if-then rules</b> — you can trace exactly WHY a passenger
          was predicted to survive or not.
        </div>
        """, unsafe_allow_html=True)

        from sklearn.tree import export_text

        depth_choice = st.slider("Tree depth", 2, 6, 4)

        @st.cache_resource
        def get_dt_rules(depth):
            dt = DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=42)
            dt.fit(X_sc, y_arr)
            return dt

        dt_rules = get_dt_rules(depth_choice)
        dt_auc   = roc_auc_score(y_arr, dt_rules.predict_proba(X_sc)[:, 1])
        dt_acc   = accuracy_score(y_arr, dt_rules.predict(X_sc))
        dt_f1    = f1_score(y_arr, dt_rules.predict(X_sc))

        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Depth",    depth_choice)
        c_m2.metric("ROC-AUC",  f"{dt_auc*100:.2f}%")
        c_m3.metric("Accuracy", f"{dt_acc*100:.2f}%")
        c_m4.metric("F1-Score", f"{dt_f1*100:.2f}%")

        # Text rules
        rules_text = export_text(
            dt_rules,
            feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURES]
        )
        st.markdown('<div class="section-header">If-Then Decision Rules</div>',
                    unsafe_allow_html=True)
        st.code(rules_text, language='text')

        # Most important splits
        st.markdown('<div class="section-header">Top Decision Splits (by Information Gain)</div>',
                    unsafe_allow_html=True)
        feat_imp_dt = pd.DataFrame({
            'Feature': [FEATURE_LABELS.get(f, f) for f in FEATURES],
            'Importance': dt_rules.feature_importances_
        }).sort_values('Importance', ascending=False)
        feat_imp_dt = feat_imp_dt[feat_imp_dt['Importance'] > 0].head(10)

        fig_dt_imp = px.bar(
            feat_imp_dt, x='Feature', y='Importance',
            color='Importance', color_continuous_scale='Oranges',
            text=feat_imp_dt['Importance'].apply(lambda x: f"{x:.4f}"),
            title=f"Decision Tree (depth={depth_choice}) — Feature Splits Used"
        )
        fig_dt_imp.update_traces(textposition='outside')
        fig_dt_imp.update_layout(coloraxis_showscale=False, xaxis_tickangle=-25)
        dark_fig(fig_dt_imp, 380)
        st.plotly_chart(fig_dt_imp, use_container_width=True)

        # Key rules summary
        st.markdown('<div class="section-header">📖 Human-Readable Survival Rules</div>',
                    unsafe_allow_html=True)
        rules_summary = [
            ("Rule 1 — Women First",   C_GREEN,
             "IF Sex = Female → Survival probability jumps to ~74%<br>"
             "IF Sex = Male   → Survival probability drops to ~19%"),
            ("Rule 2 — Class Effect",  C_BLUE,
             "IF 1st Class → +12% survival boost<br>"
             "IF 3rd Class → −15% survival penalty"),
            ("Rule 3 — Children",      C_CYAN,
             "IF Age < 12 → Evacuation priority, survival ~58%<br>"
             "Master title is strongest proxy for young boys"),
            ("Rule 4 — Wealth Signal", C_YELLOW,
             "IF Fare > £30 → Higher cabin access → better survival<br>"
             "Has_Cabin flag captures similar signal"),
            ("Rule 5 — Family Size",   C_PINK,
             "IF Family_Size ∈ [2,4] → Best survival (not alone, not too large)<br>"
             "IF Alone OR Family ≥ 5 → Lower survival"),
        ]
        for title_r, color_r, body_r in rules_summary:
            st.markdown(f"""
            <div style="background:rgba(17,24,39,0.9);border:1px solid #374151;
                        border-left:4px solid {color_r};border-radius:8px;
                        padding:14px 18px;margin:8px 0">
              <div style="font-weight:700;color:{color_r};margin-bottom:6px">{title_r}</div>
              <div style="color:#d1d5db;font-size:0.88em">{body_r}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Hyperparameter Tuning":
    st.markdown("## 🔧 Hyperparameter Tuning — GridSearchCV & RandomizedSearchCV")

    st.markdown("""
    <div class="info">
      <b>Strategy:</b><br>
      • <b>GridSearchCV</b> — exhaustive search for smaller grids (LR, SVM, DT)<br>
      • <b>RandomizedSearchCV</b> — efficient sampling for large spaces (RF, GB, ET)<br>
      • All searches use <b>5-Fold Stratified CV</b>, scored on <b>ROC-AUC</b><br>
      • ⚠️ First run takes 3–8 minutes — results are cached automatically
    </div>
    """, unsafe_allow_html=True)

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from scipy.stats import randint, uniform, loguniform
    import time as _time

    # ── Grids ────────────────────────────────────────────────────────────────
    GRIDS = {
        'Logistic Regression': {
            'type': 'grid',
            'estimator': LogisticRegression(random_state=42),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 5, 10, 50],
                'solver': ['lbfgs', 'liblinear'],
                'class_weight': ['balanced', None],
                'max_iter': [1000],
            },
        },
        'SVM': {
            'type': 'grid',
            'estimator': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 5, 10, 50],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced', None],
            },
        },
        'Decision Tree': {
            'type': 'grid',
            'estimator': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 4, 5, 6, 8, None],
                'min_samples_split': [2, 4, 8, 16],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None],
                'criterion': ['gini', 'entropy'],
            },
        },
        'Random Forest': {
            'type': 'random',
            'estimator': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': randint(50, 400),
                'max_depth': [4, 5, 6, 7, 8, None],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 8),
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'class_weight': ['balanced', 'balanced_subsample', None],
            },
            'n_iter': 50,
        },
        'Gradient Boosting': {
            'type': 'random',
            'estimator': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': randint(50, 350),
                'max_depth': randint(2, 7),
                'learning_rate': loguniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'min_samples_split': randint(2, 15),
                'max_features': ['sqrt', 'log2', None],
            },
            'n_iter': 50,
        },
        'Extra Trees': {
            'type': 'random',
            'estimator': ExtraTreesClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': randint(50, 350),
                'max_depth': [4, 5, 6, 7, 8, None],
                'min_samples_split': randint(2, 15),
                'min_samples_leaf': randint(1, 8),
                'max_features': ['sqrt', 'log2', 0.4],
                'class_weight': ['balanced', None],
            },
            'n_iter': 50,
        },
    }

    cv_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    @st.cache_resource(show_spinner=False)
    def run_tuning(_X_sc, _y):
        tuned = {}
        for name, cfg in GRIDS.items():
            if cfg['type'] == 'grid':
                search = GridSearchCV(
                    cfg['estimator'], cfg['params'],
                    cv=cv_tune, scoring='roc_auc', n_jobs=-1, refit=True
                )
            else:
                search = RandomizedSearchCV(
                    cfg['estimator'], cfg['params'],
                    n_iter=cfg['n_iter'], cv=cv_tune,
                    scoring='roc_auc', n_jobs=-1,
                    random_state=42, refit=True
                )
            search.fit(_X_sc, _y)
            # baseline
            base_scores = cross_val_score(
                cfg['estimator'], _X_sc, _y,
                cv=cv_tune, scoring='roc_auc', n_jobs=-1
            )
            tuned[name] = {
                'search':       search,
                'best_score':   search.best_score_,
                'best_params':  search.best_params_,
                'best_model':   search.best_estimator_,
                'baseline_auc': base_scores.mean(),
                'search_type':  cfg['type'],
            }
        return tuned

    sel_models_tune = st.multiselect(
        "Select models to tune (fewer = faster):",
        list(GRIDS.keys()),
        default=['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        key="tune_sel"
    )

    if not sel_models_tune:
        st.warning("Select at least one model.")
        st.stop()

    run_btn = st.button("🚀 Run Hyperparameter Tuning", type="primary", use_container_width=True)

    GRIDS_SEL = {k: GRIDS[k] for k in sel_models_tune}

    @st.cache_resource(show_spinner=False)
    def run_tuning_sel(_X_sc, _y, model_names_key):
        tuned = {}
        for name in model_names_key:
            cfg = GRIDS[name]
            if cfg['type'] == 'grid':
                search = GridSearchCV(
                    cfg['estimator'], cfg['params'],
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring='roc_auc', n_jobs=-1, refit=True
                )
            else:
                search = RandomizedSearchCV(
                    cfg['estimator'], cfg['params'],
                    n_iter=cfg['n_iter'],
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring='roc_auc', n_jobs=-1, random_state=42, refit=True
                )
            search.fit(_X_sc, _y)
            base = cross_val_score(cfg['estimator'], _X_sc, _y,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                    scoring='roc_auc', n_jobs=-1).mean()
            tuned[name] = {
                'search':      search, 'best_score': search.best_score_,
                'best_params': search.best_params_,
                'best_model':  search.best_estimator_,
                'baseline_auc': base,
                'search_type': cfg['type'],
            }
        return tuned

    cache_key = tuple(sorted(sel_models_tune))
    with st.spinner(f"⏳ Tuning {len(sel_models_tune)} models with 5-Fold CV... (~2–5 min)"):
        tuned_results = run_tuning_sel(X_sc, y_arr, cache_key)

    # ── Results ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Tuning Results — Before vs After</div>',
                unsafe_allow_html=True)

    rows_t = []
    for name, res in tuned_results.items():
        delta = res['best_score'] - res['baseline_auc']
        rows_t.append({
            'Model':       name,
            'Search Type': '🔲 Grid' if res['search_type']=='grid' else '🎲 Random',
            'Baseline AUC': f"{res['baseline_auc']*100:.2f}%",
            'Tuned AUC':   f"{res['best_score']*100:.2f}%",
            'Δ AUC':       f"{delta*100:+.2f}%",
            'Improved':    '✅' if delta > 0.001 else ('→' if abs(delta)<=0.001 else '❌'),
            '_delta':      delta,
            '_tuned':      res['best_score'],
            '_base':       res['baseline_auc'],
        })

    rows_t_df = pd.DataFrame(rows_t).sort_values('_tuned', ascending=False)
    st.dataframe(rows_t_df.drop(columns=['_delta','_tuned','_base']),
                 use_container_width=True, hide_index=True)

    # Visual comparison
    c1, c2 = st.columns(2)
    with c1:
        fig_cmp = go.Figure()
        names_t  = rows_t_df['Model'].tolist()
        base_t   = rows_t_df['_base'].tolist()
        tuned_t  = rows_t_df['_tuned'].tolist()
        x_t      = np.arange(len(names_t))
        fig_cmp.add_trace(go.Bar(name='Baseline', x=names_t, y=base_t,
                                   marker_color=C_BLUE, opacity=0.55,
                                   text=[f"{v*100:.2f}%" for v in base_t], textposition='inside'))
        fig_cmp.add_trace(go.Bar(name='Tuned', x=names_t, y=tuned_t,
                                   marker_color=C_GREEN, opacity=0.88,
                                   text=[f"{v*100:.2f}%" for v in tuned_t], textposition='inside'))
        fig_cmp.update_layout(barmode='group', yaxis_range=[0.6, 1.0],
                               yaxis_tickformat='.0%',
                               title='Baseline vs Tuned AUC', xaxis_tickangle=-20)
        dark_fig(fig_cmp, 380)
        st.plotly_chart(fig_cmp, use_container_width=True)

    with c2:
        deltas_t = rows_t_df['_delta'].tolist()
        fig_d = go.Figure(go.Bar(
            x=names_t, y=[d*100 for d in deltas_t],
            marker_color=[C_GREEN if d > 0 else C_RED for d in deltas_t],
            text=[f"{d*100:+.2f}%" for d in deltas_t], textposition='outside',
            opacity=0.88
        ))
        fig_d.add_hline(y=0, line_color='white', line_width=1.5)
        fig_d.update_layout(title='Tuning Gain (%pts)', yaxis_title='AUC improvement',
                             xaxis_tickangle=-20)
        dark_fig(fig_d, 380)
        st.plotly_chart(fig_d, use_container_width=True)

    # ── Best params per model ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Best Hyperparameters Found</div>',
                unsafe_allow_html=True)
    for name, res in tuned_results.items():
        color = MODEL_COLORS.get(name, C_BLUE)
        with st.expander(f"{name} — Best AUC: {res['best_score']*100:.2f}% "
                         f"(Δ {(res['best_score']-res['baseline_auc'])*100:+.2f}%)"):
            params_display = {k: str(v) for k, v in res['best_params'].items()}
            p_df = pd.DataFrame(list(params_display.items()),
                                 columns=['Parameter', 'Best Value'])
            st.dataframe(p_df, use_container_width=True, hide_index=True)

            # Search CV results scatter
            if hasattr(res['search'], 'cv_results_'):
                cv_res_df = pd.DataFrame(res['search'].cv_results_)
                if 'mean_test_score' in cv_res_df.columns:
                    fig_scatter = px.histogram(
                        cv_res_df, x='mean_test_score', nbins=25,
                        color_discrete_sequence=[color],
                        title=f"Distribution of AUC Scores Across All Combinations Tried"
                    )
                    fig_scatter.add_vline(x=res['best_score'], line_dash='dash',
                                           line_color='white',
                                           annotation_text=f"Best={res['best_score']:.4f}")
                    dark_fig(fig_scatter, 280)
                    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Search size info ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 Search Space Summary</div>',
                unsafe_allow_html=True)
    info_rows = []
    for name, res in tuned_results.items():
        n_comb = len(res['search'].cv_results_['mean_test_score'])
        info_rows.append({
            'Model': name,
            'Type': 'GridSearchCV' if res['search_type']=='grid' else 'RandomizedSearchCV',
            'Combinations tried': n_comb,
            'Total CV fits': n_comb * 5,
            'Best AUC': f"{res['best_score']*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight">
      💡 <b>RandomizedSearchCV</b> explores ~50 combinations vs potentially thousands in GridSearchCV,
      but typically recovers 90%+ of the best performance — ideal for Random Forest and Gradient Boosting
      where the search space is huge (millions of combinations).
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: KAGGLE SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏅 Kaggle Submission":
    st.markdown("## 🏅 Kaggle Submission Generator")

    st.markdown("""
    <div class="info">
      Upload the Kaggle <b>test.csv</b> (418 passengers, no Survived column)
      and download a ready-to-submit <b>submission.csv</b> using your best trained model.
    </div>
    """, unsafe_allow_html=True)

    c_up, c_cfg = st.columns([1, 1])

    with c_up:
        st.markdown('<div class="section-header">📁 Upload Test File</div>',
                    unsafe_allow_html=True)
        test_file = st.file_uploader("Upload test.csv", type=['csv'])
        st.markdown("""
        <div style="font-size:0.8em;color:#6b7280;margin-top:8px">
          Download from:
          <a href="https://www.kaggle.com/c/titanic/data" target="_blank"
             style="color:#6366f1">kaggle.com/c/titanic/data</a><br>
          No file? We'll generate a demo prediction using a synthetic test set.
        </div>
        """, unsafe_allow_html=True)

    with c_cfg:
        st.markdown('<div class="section-header">⚙️ Submission Settings</div>',
                    unsafe_allow_html=True)
        sub_model_name = st.selectbox("Model", list(trained.keys()),
                                       index=list(trained.keys()).index(best_name))
        threshold_sub  = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.01)
        include_proba  = st.checkbox("Include probability column (analysis only)", True)

    # ── Load / generate test set ──────────────────────────────────────────────
    if test_file is not None:
        test_raw = pd.read_csv(test_file)
        st.success(f"✅ Uploaded: {len(test_raw)} passengers")
    else:
        st.markdown('<div class="warning">⚠️ No file uploaded — using synthetic 418-passenger demo set.</div>',
                    unsafe_allow_html=True)
        np.random.seed(99)
        n_t = 418
        test_raw = pd.DataFrame({
            'PassengerId': range(892, 892+n_t),
            'Pclass':  np.random.choice([1,2,3], n_t, p=[0.25,0.22,0.53]),
            'Name':    [f"Passenger{i}, Mr. Demo" for i in range(n_t)],
            'Sex':     np.random.choice(['male','female'], n_t, p=[0.64,0.36]),
            'Age':     np.where(np.random.rand(n_t)<0.2, np.nan,
                                np.random.normal(29,13,n_t).clip(1,80)),
            'SibSp':   np.random.choice([0,1,2], n_t, p=[0.61,0.27,0.12]),
            'Parch':   np.random.choice([0,1,2], n_t, p=[0.73,0.18,0.09]),
            'Ticket':  [f"PC{i:04d}" for i in range(n_t)],
            'Fare':    np.random.exponential(33, n_t).clip(0, 512),
            'Cabin':   [np.nan if np.random.rand()<0.77 else f"C{i:02d}" for i in range(n_t)],
            'Embarked':np.random.choice(['S','C','Q'], n_t, p=[0.72,0.19,0.09]),
        })

    # ── Feature engineering on test ───────────────────────────────────────────
    eng_test = engineer_features(test_raw)
    X_test_df = X.copy().iloc[:1]  # just to get columns

    # Prepare test features (impute with train medians)
    X_test_raw = eng_test[FEATURES].copy()
    for pclass in [1, 2, 3]:
        for sex in ['male', 'female']:
            mask_all   = (eng_df['Pclass']==pclass) & (eng_df['Sex']==sex)
            median_age = eng_df.loc[mask_all & eng_df['Age'].notna(), 'Age'].median()
            test_mask  = (eng_test['Pclass']==pclass) & (eng_test['Sex']==sex) & eng_test['Age'].isna()
            X_test_raw.loc[test_mask, 'Age'] = median_age

    X_test_raw['Age']            = X_test_raw['Age'].fillna(X['Age'].median())
    X_test_raw['Fare']           = X_test_raw['Fare'].fillna(X['Fare'].median())
    X_test_raw['Fare_Per_Person'] = X_test_raw['Fare_Per_Person'].fillna(X['Fare_Per_Person'].median())
    X_test_raw = X_test_raw.astype(float)

    # Scale using TRAIN scaler
    X_test_sc = scaler.transform(X_test_raw)

    # Predict
    sub_model   = trained[sub_model_name]
    test_proba  = sub_model.predict_proba(X_test_sc)[:, 1]
    test_preds  = (test_proba >= threshold_sub).astype(int)

    # ── Submission file ────────────────────────────────────────────────────────
    if include_proba:
        sub_df = pd.DataFrame({
            'PassengerId': test_raw['PassengerId'],
            'Survived':    test_preds,
            'Probability': test_proba.round(4),
        })
    else:
        sub_df = pd.DataFrame({
            'PassengerId': test_raw['PassengerId'],
            'Survived':    test_preds,
        })

    kaggle_df = pd.DataFrame({
        'PassengerId': test_raw['PassengerId'],
        'Survived':    test_preds,
    })

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    surv_n  = int(test_preds.sum())
    dead_n  = int((1 - test_preds).sum())
    surv_r  = test_preds.mean() * 100
    hi_conf = int((test_proba > 0.8).sum() + (test_proba < 0.2).sum())
    uncert  = int(((test_proba >= 0.4) & (test_proba <= 0.6)).sum())

    for col, (label, val, color) in zip(
        [kc1, kc2, kc3, kc4, kc5],
        [("Total",        f"{len(test_preds)}",   C_BLUE),
         ("Survived",     f"{surv_n}",            C_GREEN),
         ("Not Survived", f"{dead_n}",            C_RED),
         ("Survival Rate",f"{surv_r:.1f}%",       C_YELLOW),
         ("Uncertain",    f"{uncert}",             C_PINK)]):
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:{color}">{val}</div>
          <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download ───────────────────────────────────────────────────────────────
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown('<div class="section-header">⬇️ Kaggle submission.csv</div>',
                    unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download submission.csv (Kaggle format)",
            data=kaggle_df.to_csv(index=False).encode(),
            file_name="submission.csv", mime="text/csv",
            use_container_width=True, type="primary"
        )
        st.markdown(f"""
        <div class="insight">
          ✅ <b>Kaggle-ready format</b><br>
          Columns: PassengerId + Survived (0/1)<br>
          Rows: {len(kaggle_df)}<br>
          Model: {sub_model_name}<br>
          Threshold: {threshold_sub}
        </div>
        """, unsafe_allow_html=True)

    with dc2:
        st.markdown('<div class="section-header">⬇️ Full analysis CSV (with probabilities)</div>',
                    unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download submission_with_proba.csv",
            data=sub_df.to_csv(index=False).encode(),
            file_name="submission_with_proba.csv", mime="text/csv",
            use_container_width=True
        )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Prediction Analysis</div>',
                unsafe_allow_html=True)
    ca1, ca2, ca3 = st.columns(3)

    with ca1:
        fig_dist = px.histogram(pd.DataFrame({'P(Survive)': test_proba}),
                                 x='P(Survive)', nbins=40,
                                 color_discrete_sequence=[C_BLUE],
                                 title='Survival Probability Distribution')
        fig_dist.add_vline(x=threshold_sub, line_dash='dash', line_color=C_RED,
                            annotation_text=f"Threshold={threshold_sub}")
        fig_dist.add_vline(x=test_proba.mean(), line_dash='dot', line_color=C_YELLOW,
                            annotation_text=f"Mean={test_proba.mean():.2f}")
        dark_fig(fig_dist, 330)
        st.plotly_chart(fig_dist, use_container_width=True)

    with ca2:
        pclass_col = test_raw['Pclass'].values
        sex_col    = test_raw['Sex'].values
        pclass_surv_r = {}
        for cls in [1, 2, 3]:
            mask = pclass_col == cls
            pclass_surv_r[f'{cls}{"st" if cls==1 else "nd" if cls==2 else "rd"} Class'] = \
                test_preds[mask].mean() * 100

        fig_cls = px.bar(
            x=list(pclass_surv_r.keys()), y=list(pclass_surv_r.values()),
            color=list(pclass_surv_r.values()),
            color_continuous_scale='RdYlGn', range_color=[0, 100],
            text=[f"{v:.1f}%" for v in pclass_surv_r.values()],
            title='Predicted Survival Rate by Class'
        )
        fig_cls.update_traces(textposition='outside')
        fig_cls.update_layout(coloraxis_showscale=False, yaxis_range=[0, 110])
        dark_fig(fig_cls, 330)
        st.plotly_chart(fig_cls, use_container_width=True)

    with ca3:
        sex_data = {
            s: test_preds[sex_col == s].mean() * 100
            for s in ['female', 'male']
        }
        fig_sex = px.bar(
            x=list(sex_data.keys()), y=list(sex_data.values()),
            color=list(sex_data.keys()),
            color_discrete_map={'female': C_PINK, 'male': C_BLUE},
            text=[f"{v:.1f}%" for v in sex_data.values()],
            title='Predicted Survival Rate by Sex'
        )
        fig_sex.update_traces(textposition='outside')
        fig_sex.update_layout(yaxis_range=[0, 110], showlegend=False)
        dark_fig(fig_sex, 330)
        st.plotly_chart(fig_sex, use_container_width=True)

    # Preview table
    st.markdown('<div class="section-header">👀 Preview — First 20 Predictions</div>',
                unsafe_allow_html=True)
    preview = sub_df.head(20).copy()
    preview['Verdict'] = preview['Survived'].map({1: '✅ Survived', 0: '❌ Not Survived'})
    if 'Probability' in preview.columns:
        preview['Confidence'] = preview['Probability'].apply(
            lambda p: '🟢 High' if p > 0.75 or p < 0.25 else ('🟡 Medium' if p > 0.6 or p < 0.4 else '🔴 Uncertain')
        )
    st.dataframe(preview, use_container_width=True, hide_index=True)

    # Kaggle instructions
    st.markdown('<div class="section-header">📋 How to Submit to Kaggle</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="model-card">
      <div style="display:grid;grid-template-columns:auto 1fr;gap:12px 20px;font-size:0.88em">
        <div style="color:#6366f1;font-weight:700">Step 1</div>
        <div>Go to <a href="https://www.kaggle.com/c/titanic" target="_blank"
             style="color:#6366f1">kaggle.com/c/titanic</a> and join the competition</div>
        <div style="color:#6366f1;font-weight:700">Step 2</div>
        <div>Click <b>"Submit Predictions"</b></div>
        <div style="color:#6366f1;font-weight:700">Step 3</div>
        <div>Upload the downloaded <b>submission.csv</b></div>
        <div style="color:#6366f1;font-weight:700">Step 4</div>
        <div>Kaggle scores on the hidden 50% of test set — top scores ~85%+</div>
      </div>
      <div style="margin-top:14px;font-size:0.82em;color:#6b7280">
        ⚠️ Kaggle leaderboard uses <b>Accuracy</b> (not AUC) for Titanic.
        Typical scores: Random=62%, Logistic Reg=~77%, Tuned RF=~80–83%, Best published=~85%
      </div>
    </div>
    """, unsafe_allow_html=True)

