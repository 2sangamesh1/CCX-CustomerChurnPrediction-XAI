"""
=============================================================================
MULTIMODAL CHURN PREDICTION WITH LLM-ENHANCED FEATURES:
AN INTERPRETABLE AND PRESCRIPTIVE FRAMEWORK
=============================================================================
Research Implementation — Full Pipeline (GPU-Optimized)
=============================================================================

Required packages (pip install):
    pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
    xgboost lightgbm shap lime sentence-transformers pulp joblib wordcloud

For GPU acceleration:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install xgboost  # (built with CUDA support)
    pip install cuml cudf  # (optional — RAPIDS for sklearn on GPU)

Optional (recommended for faster tuning):
    pip install optuna

Run:
    python churn_research_pipeline.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
import joblib
import torch
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    RandomizedSearchCV, learning_curve, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD
from scipy import stats
import shap
import xgboost as xgb
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

warnings.filterwarnings("ignore")

# ---------- optional imports ----------
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import lime, lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    from wordcloud import WordCloud
    HAS_WC = True
except ImportError:
    HAS_WC = False

# ★ Optuna for smart Bayesian tuning with early pruning
try:
    import optuna
    from optuna.integration import XGBoostPruningCallback
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ★ HalvingRandomSearchCV — sklearn's successive halving
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa
    from sklearn.model_selection import HalvingRandomSearchCV
    HAS_HALVING = True
except ImportError:
    HAS_HALVING = False

# ---------- GPU detection ----------
def detect_gpu():
    """Detect available GPU and return device info."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
        "device": "cpu",
        "vram_gb": 0,
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device"] = "cuda"
        info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  🟢 GPU detected: {info['device_name']} "
              f"({info['vram_gb']:.1f} GB VRAM)")
    else:
        print("  🟡 No GPU detected — running on CPU")
        print("    To enable GPU:")
        print("    pip install torch --index-url "
              "https://download.pytorch.org/whl/cu118")
    return info

GPU_INFO = detect_gpu()


# ====================================================================
# SECTION 1 — CENTRAL CONFIGURATION
# ====================================================================
class Config:
    STRUCTURED_URL = (
        "https://raw.githubusercontent.com/YBIFoundation/"
        "Dataset/main/TelecomCustomerChurn.csv"
    )
    TEXT_URL = (
        "https://raw.githubusercontent.com/aws-samples/"
        "churn-prediction-with-text-and-interpretability/"
        "refs/heads/main/data/text.csv"
    )
    OUTPUT      = "research_outputs"
    FIG_DIR     = os.path.join(OUTPUT, "figures")
    MODEL_DIR   = os.path.join(OUTPUT, "models")
    RESULT_DIR  = os.path.join(OUTPUT, "results")

    SEED        = 42
    N_SPLITS    = 5
    TEST_SIZE   = 0.20
    USE_SMOTE   = False

    EMB_MODELS  = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    PCA_N       = 50
    USE_PCA     = True
    TFIDF_MAX   = 50

    # ---- GPU-specific settings ----
    USE_GPU           = GPU_INFO["cuda_available"]
    EMB_BATCH_SIZE    = 256 if GPU_INFO["cuda_available"] else 64
    EMB_DEVICE        = "cuda" if GPU_INFO["cuda_available"] else "cpu"

    # ============================================================
    # ★ TUNING STRATEGY SELECTION
    # ============================================================
    # Options: "optuna"    — Bayesian optimization with pruning (fastest)
    #          "halving"   — successive halving (fast, no extra deps)
    #          "randomized"— RandomizedSearchCV (moderate)
    #          "grid"      — exhaustive GridSearchCV (slowest)
    # ============================================================
    TUNING_STRATEGY = (
        "optuna" if HAS_OPTUNA else
        "halving" if HAS_HALVING else
        "randomized"
    )

    # Optuna settings
    OPTUNA_N_TRIALS    = 50       # total trials per model
    OPTUNA_TIMEOUT     = 120      # max seconds per model (None = no limit)
    OPTUNA_PRUNING     = True     # early-stop bad trials

    # RandomizedSearchCV settings
    N_ITER_RANDOM      = 30       # random combos to try

    # Inner CV folds for tuning (fewer = faster)
    TUNE_CV_FOLDS      = 3

    # ★ Which models to tune (skip slow ones you don't need)
    TUNE_MODELS = ["LR", "RF", "XGB", "MLP"]  # add "LGBM" if available
    if HAS_LGBM:
        TUNE_MODELS.append("LGBM")

    GRIDS = {
        "LR": {
            "C": [0.01, 0.1, 1, 10],
            "max_iter": [3000],
            "class_weight": [None, "balanced"],
        },
        "RF": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
        "XGB": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "scale_pos_weight": [1, 2.8, 5, 10],
        },
        "MLP": {
            "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
            "alpha": [1e-4, 1e-3],
            "max_iter": [500],
        },
    }
    if HAS_LGBM:
        GRIDS["LGBM"] = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 50],
            "is_unbalance": [True, False],
        }

    BUDGET  = 5000
    FIG_DPI = 300

    @classmethod
    def setup(cls):
        for d in [cls.OUTPUT, cls.FIG_DIR, cls.MODEL_DIR, cls.RESULT_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def xgb_gpu_params(cls):
        """Return XGBoost params for GPU training."""
        if cls.USE_GPU:
            return {
                "tree_method": "hist",
                "device": "cuda",
            }
        return {"tree_method": "hist"}

    @classmethod
    def lgbm_gpu_params(cls):
        """Return LightGBM params for GPU training."""
        if cls.USE_GPU:
            return {"device": "gpu", "gpu_use_dp": False}
        return {}


# ====================================================================
# SECTION 2 — EXPERIMENT LOGGER
# ====================================================================
class Logger:
    def __init__(self):
        self.log = {"ts": datetime.now().isoformat(), "sections": {}}

    def add(self, section, key, val):
        self.log["sections"].setdefault(section, {})[key] = val

    def save(self, path):
        def _conv(o):
            if isinstance(o, (np.integer,)):   return int(o)
            if isinstance(o, (np.floating,)):  return float(o)
            if isinstance(o, np.ndarray):      return o.tolist()
            if isinstance(o, pd.DataFrame):    return o.to_dict()
            return str(o)
        with open(path, "w") as f:
            json.dump(self.log, f, default=_conv, indent=2)
        print(f"Log → {path}")


# ====================================================================
# SECTION 3 — DATA PROCESSOR & FEATURE ENGINEERING
# ====================================================================
class DataProcessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.df = None
        self.ohe_cols = []
        self.eng_cols = []

    def load(self):
        print("=" * 65)
        print("LOADING DATA")
        print("=" * 65)
        try:
            self.df = pd.read_csv(self.cfg.STRUCTURED_URL)
            print(f"Structured loaded : {self.df.shape}")
        except Exception as e:
            print(f"URL failed ({e}); generating synthetic data")
            self.df = self._synth_struct()

        if self.cfg.TEXT_URL:
            try:
                tdf = pd.read_csv(self.cfg.TEXT_URL)
                tcols = [c for c in tdf.columns
                         if any(k in c.lower()
                                for k in ("text", "chat", "log"))]
                if tcols and len(tdf) == len(self.df):
                    self.df["chat_log"] = tdf[tcols[0]].values
                else:
                    self._add_synth_text()
            except Exception:
                self._add_synth_text()
        else:
            self._add_synth_text()
        print(f"Final shape        : {self.df.shape}")
        return self.df

    def _synth_struct(self):
        rng = np.random.RandomState(self.cfg.SEED)
        n = 1000
        return pd.DataFrame({
            "customerID": [f"C{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[.84, .16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n, p=[.3, .7]),
            "tenure": rng.randint(0, 73, n),
            "PhoneService": rng.choice(["Yes", "No"], n, p=[.9, .1]),
            "MultipleLines": rng.choice(
                ["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(
                ["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(
                ["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(
                ["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(
                ["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(
                ["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(
                ["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(
                ["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"],
                n, p=[.55, .21, .24]),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)",
                 "Credit card (automatic)"], n),
            "MonthlyCharges": np.round(rng.uniform(18, 120, n), 2),
            "TotalCharges": np.round(rng.uniform(18, 8700, n), 2),
            "Churn": rng.choice(["Yes", "No"], n, p=[.27, .73]),
        })

    def _add_synth_text(self):
        rng = np.random.RandomState(self.cfg.SEED)

        tenure_snip = {
            "short": [
                "Customer: I just signed up recently.",
                "Customer: I'm a new customer here.",
                "Customer: I started my plan a few weeks ago.",
            ],
            "mid": [
                "Customer: I've been with you for about a year now.",
                "Customer: I've had this plan for a while.",
                "Customer: I've been a customer for some time.",
            ],
            "long": [
                "Customer: I've been a loyal customer for many years.",
                "Customer: I've used your service for a long time.",
                "Customer: I've been here since the early days.",
            ],
        }
        contract_snip = {
            "Month-to-month": [
                "Agent: You are on a month-to-month plan. "
                "Customer: Yes, I like flexibility.",
                "Agent: Your plan renews monthly. "
                "Customer: That's correct.",
            ],
            "One year": [
                "Agent: You have a one-year contract. "
                "Customer: Right, it made sense for me.",
                "Agent: Your annual contract is active. "
                "Customer: Good to know.",
            ],
            "Two year": [
                "Agent: You're on a two-year agreement. "
                "Customer: Yes, I wanted the best rate.",
                "Agent: Your long-term contract is in place. "
                "Customer: That's what I signed up for.",
            ],
        }
        internet_snip = {
            "Fiber optic": [
                "Customer: I have fiber optic internet. "
                "Agent: Great, that's our fastest tier.",
            ],
            "DSL": [
                "Customer: I'm on DSL internet. "
                "Agent: Let me check your speed options.",
            ],
            "No": [
                "Customer: I don't have internet service with you. "
                "Agent: Understood.",
            ],
        }
        charge_snip = {
            "low": [
                "Customer: My bill is quite affordable.",
                "Customer: The monthly charge seems reasonable.",
            ],
            "mid": [
                "Customer: My bill is about average I think.",
                "Customer: The monthly charges are moderate.",
            ],
            "high": [
                "Customer: My monthly bill is quite high.",
                "Customer: I'm paying a lot each month.",
            ],
        }
        random_filler = [
            " Agent: Is there anything else I can help with?",
            " Agent: Thank you for calling.",
            " Agent: We appreciate your patience.",
            " Customer: Thanks for the info.",
            " Customer: Okay, I'll think about it.",
            " Agent: Happy to assist.",
            " Customer: That's all for now.",
            " Agent: Let me know if you need anything.",
        ]

        t_col = ("tenure" if "tenure" in self.df.columns else "Tenure")
        c_col = ("Contract" if "Contract" in self.df.columns
                 else "contract")
        i_col = ("InternetService" if "InternetService" in self.df.columns
                 else "internetservice")
        m_col = ("MonthlyCharges" if "MonthlyCharges" in self.df.columns
                 else "monthlycharges")

        logs = []
        for _, row in self.df.iterrows():
            parts = []
            ten = float(row.get(t_col, 0) or 0)
            if ten <= 12:
                parts.append(rng.choice(tenure_snip["short"]))
            elif ten <= 36:
                parts.append(rng.choice(tenure_snip["mid"]))
            else:
                parts.append(rng.choice(tenure_snip["long"]))

            cval = str(row.get(c_col, "Month-to-month"))
            parts.append(rng.choice(
                contract_snip.get(cval, contract_snip["Month-to-month"])))

            ival = str(row.get(i_col, "No"))
            parts.append(rng.choice(
                internet_snip.get(ival, internet_snip["No"])))

            mc = float(row.get(m_col, 0) or 0)
            if mc < 40:
                parts.append(rng.choice(charge_snip["low"]))
            elif mc < 80:
                parts.append(rng.choice(charge_snip["mid"]))
            else:
                parts.append(rng.choice(charge_snip["high"]))

            parts.append(rng.choice(random_filler))
            logs.append(" ".join(parts))

        self.df["chat_log"] = logs

    def engineer(self):
        print("\n" + "=" * 65)
        print("FEATURE ENGINEERING")
        print("=" * 65)
        self.df.columns = (self.df.columns.str.lower()
                           .str.replace(" ", ""))

        if self.df["churn"].dtype == object:
            self.df["churn"] = self.df["churn"].map({"Yes": 1, "No": 0})

        if "totalcharges" in self.df.columns:
            self.df["totalcharges"] = pd.to_numeric(
                self.df["totalcharges"], errors="coerce")
            self.df["totalcharges_missing"] = (
                self.df["totalcharges"].isna().astype(int))
            self.eng_cols.append("totalcharges_missing")
            self.df["totalcharges"].fillna(
                self.df["totalcharges"].median(), inplace=True)

        mc_max = self.df["monthlycharges"].max()
        if mc_max > 0:
            self.df["engagementscore"] = (
                self.df["tenure"]
                * (self.df["monthlycharges"] / mc_max))
            self.eng_cols.append("engagementscore")

        if "techsupport" in self.df.columns:
            mapping = {"Yes": 1.0, "No internet service": 0.5}
            self.df["servicesatisfaction"] = (
                self.df["techsupport"].apply(
                    lambda x: mapping.get(x, 0.0)
                    if isinstance(x, str) else x))
            self.eng_cols.append("servicesatisfaction")

        self.df["pricesensitivity"] = (
            self.df["monthlycharges"] / (self.df["tenure"] + 1))
        self.eng_cols.append("pricesensitivity")

        if "contract" in self.df.columns:
            cmap = {"Month-to-month": 1, "One year": 5, "Two year": 10}
            if self.df["contract"].dtype == object:
                self.df["loyaltyscore"] = (
                    self.df["contract"].map(cmap).fillna(1)
                    * self.df["tenure"])
            else:
                self.df["loyaltyscore"] = self.df["tenure"]
            self.eng_cols.append("loyaltyscore")

        if "totalcharges" in self.df.columns:
            self.df["avgmonthlyspend"] = (
                self.df["totalcharges"] / (self.df["tenure"] + 1))
            self.eng_cols.append("avgmonthlyspend")

        svc = [
            "phoneservice", "multiplelines", "internetservice",
            "onlinesecurity", "onlinebackup", "deviceprotection",
            "techsupport", "streamingtv", "streamingmovies",
        ]
        svc_exist = [c for c in svc if c in self.df.columns]
        if svc_exist:
            self.df["servicecount"] = self.df[svc_exist].apply(
                lambda r: sum(1 for v in r if v in ("Yes", 1)), axis=1)
            self.eng_cols.append("servicecount")

        self.df["tenuregroup"] = pd.cut(
            self.df["tenure"], bins=[0, 12, 24, 48, 60, 72],
            labels=[1, 2, 3, 4, 5], include_lowest=True
        ).astype(float).fillna(1)
        self.eng_cols.append("tenuregroup")

        if "contract" in self.df.columns:
            cval_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            if self.df["contract"].dtype == object:
                cval = self.df["contract"].map(cval_map).fillna(0)
            else:
                cval = self.df["contract"]
            self.df["contract_value"] = (
                self.df["monthlycharges"] * (3 - cval))
            self.eng_cols.append("contract_value")

        if "totalcharges" in self.df.columns:
            expected = self.df["totalcharges"] / (self.df["tenure"] + 1)
            self.df["charge_ratio"] = (
                self.df["monthlycharges"] / (expected + 1))
            self.eng_cols.append("charge_ratio")

        if ("streamingtv" in self.df.columns
                and "streamingmovies" in self.df.columns):
            stv = self.df["streamingtv"].apply(
                lambda x: 1 if x == "Yes" or x == 1 else 0)
            smv = self.df["streamingmovies"].apply(
                lambda x: 1 if x == "Yes" or x == 1 else 0)
            self.df["streaming_bundle"] = stv + smv
            self.eng_cols.append("streaming_bundle")

        sec_cols = ["onlinesecurity", "onlinebackup", "deviceprotection"]
        sec_exist = [c for c in sec_cols if c in self.df.columns]
        if sec_exist:
            self.df["security_bundle"] = self.df[sec_exist].apply(
                lambda r: sum(1 for v in r if v in ("Yes", 1)), axis=1)
            self.eng_cols.append("security_bundle")

        if "techsupport" in self.df.columns:
            no_support = self.df["techsupport"].apply(
                lambda x: 1 if x in ("No", 0) else 0)
            self.df["nosupport_highcharge"] = (
                no_support * self.df["monthlycharges"])
            self.eng_cols.append("nosupport_highcharge")

        self.df["tenure_x_monthly"] = (
            self.df["tenure"] * self.df["monthlycharges"])
        self.eng_cols.append("tenure_x_monthly")

        print(f"  Engineered cols : {self.eng_cols}")

        cats = [
            c for c in self.df.select_dtypes(include="object").columns
            if c not in ("customerid", "chat_log")
        ]
        if cats:
            before_cols = set(self.df.columns)
            self.df = pd.get_dummies(
                self.df, columns=cats, drop_first=True, dtype=int)
            self.ohe_cols = [
                c for c in self.df.columns if c not in before_cols]
        print(f"  One-hot encoded : {len(cats)} cols "
              f"→ {len(self.ohe_cols)} dummies")
        return self.df

    def build_feature_sets(self, emb_dfs=None, tfidf_df=None):
        drop = {"customerid", "chat_log", "churn"}
        base = [c for c in self.df.columns if c not in drop]
        struct_only = [c for c in base if c not in self.eng_cols]

        fs = {}
        fs["A1_Structured"] = self.df[struct_only].copy()
        fs["A2_Struct+Eng"] = self.df[base].copy()

        if emb_dfs:
            for en, edf in emb_dfs.items():
                fs[f"A3_Struct+{en}"] = pd.concat(
                    [self.df[struct_only].reset_index(drop=True),
                     edf.reset_index(drop=True)], axis=1)
                fs[f"A4_Full+{en}"] = pd.concat(
                    [self.df[base].reset_index(drop=True),
                     edf.reset_index(drop=True)], axis=1)

        if tfidf_df is not None:
            fs["A5_Struct+Eng+TFIDF"] = pd.concat(
                [self.df[base].reset_index(drop=True),
                 tfidf_df.reset_index(drop=True)], axis=1)
        return fs


# ====================================================================
# SECTION 4 — EDA ENGINE
# ====================================================================
class EDA:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, df):
        print("\n" + "=" * 65)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 65)
        self._target(df)
        self._num_dist(df)
        self._corr(df)
        self._feat_target(df)
        self._text(df)
        print("  EDA figures saved.")

    @staticmethod
    def _safe_churn(df):
        d = df.copy()
        if "churn" not in d.columns:
            if "Churn" in d.columns:
                d.columns = d.columns.str.lower()
        if d["churn"].dtype == object:
            d["churn"] = d["churn"].map({"Yes": 1, "No": 0})
        return d

    def _target(self, df):
        d = self._safe_churn(df)
        if "churn" not in d.columns:
            return
        fig, ax = plt.subplots(figsize=(5, 4))
        cts = d["churn"].value_counts().sort_index()
        ax.bar(["No Churn", "Churn"], cts.values,
               color=["#2ecc71", "#e74c3c"])
        for i, v in enumerate(cts.values):
            ax.text(i, v + 5, f"{v}  ({v/len(d)*100:.1f}%)", ha="center")
        ax.set_title("Target Distribution")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR, "eda_target.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

    def _num_dist(self, df):
        d = self._safe_churn(df)
        nums = [c for c in d.select_dtypes(include=np.number).columns
                if c != "churn"][:9]
        if not nums:
            return
        nr = (len(nums) + 2) // 3
        fig, axes = plt.subplots(nr, 3, figsize=(15, 4 * nr))
        axes = axes.flatten()
        for i, c in enumerate(nums):
            for cv, col, lab in [(0, "#2ecc71", "No"),
                                 (1, "#e74c3c", "Yes")]:
                axes[i].hist(
                    d.loc[d["churn"] == cv, c].dropna(),
                    bins=30, alpha=.55, color=col, label=lab)
            axes[i].set_title(c, fontsize=9)
            axes[i].legend(fontsize=7)
        for j in range(len(nums), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Feature Distributions by Churn", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 "eda_distributions.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

    def _corr(self, df):
        d = self._safe_churn(df)
        num = d.select_dtypes(include=np.number)
        if num.shape[1] < 2:
            return
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(num.corr(), dtype=bool))
        sns.heatmap(num.corr(), mask=mask, cmap="RdBu_r", center=0,
                    square=True, linewidths=.4, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR, "eda_corr.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

    def _feat_target(self, df):
        d = self._safe_churn(df)
        cols = [c for c in ["tenure", "monthlycharges", "totalcharges",
                            "contract"] if c in d.columns]
        if not cols:
            return
        fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
        if len(cols) == 1:
            axes = [axes]
        for i, c in enumerate(cols):
            if d[c].nunique() <= 5:
                pd.crosstab(d[c], d["churn"], normalize="index").plot(
                    kind="bar", stacked=True, ax=axes[i],
                    color=["#2ecc71", "#e74c3c"])
            else:
                d[c] = pd.to_numeric(d[c], errors="coerce")
                d_clean = d[[c, "churn"]].dropna(subset=[c])
                if d_clean[c].empty:
                    axes[i].text(
                        0.5, 0.5, "non-numeric", ha="center",
                        va="center", transform=axes[i].transAxes)
                else:
                    d_clean.boxplot(column=c, by="churn", ax=axes[i])
            axes[i].set_title(c)
        plt.suptitle("Features vs Churn", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 "eda_feat_target.png"),
                    dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close()

    def _text(self, df):
        d = self._safe_churn(df)
        if "chat_log" not in d.columns:
            return
        d["_tl"] = d["chat_log"].astype(str).str.len()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for cv, col, lab in [(0, "#2ecc71", "No"),
                             (1, "#e74c3c", "Yes")]:
            axes[0].hist(
                d.loc[d["churn"] == cv, "_tl"],
                bins=30, alpha=.55, color=col, label=lab)
        axes[0].set_title("Text Length by Churn")
        axes[0].legend()
        if HAS_WC:
            txt = " ".join(
                d.loc[d["churn"] == 1, "chat_log"].astype(str))
            wc = WordCloud(
                width=400, height=200, background_color="white",
                colormap="Reds").generate(txt)
            axes[1].imshow(wc, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title("Word Cloud — Churners")
        else:
            axes[1].text(.5, .5, "wordcloud not installed",
                         ha="center", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR, "eda_text.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()
        d.drop("_tl", axis=1, inplace=True)


# ====================================================================
# SECTION 5 — EMBEDDING GENERATOR (GPU-accelerated)
# ====================================================================
class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pcas = {}

    def embed_all(self, text_series):
        print("\n" + "=" * 65)
        print("GENERATING EMBEDDINGS"
              + (" (GPU)" if self.cfg.USE_GPU else " (CPU)"))
        print("=" * 65)
        clean = text_series.astype(str).fillna("")
        out = {}
        for mname in self.cfg.EMB_MODELS:
            print(f"  {mname} …")
            t_start = time.time()
            try:
                model = SentenceTransformer(
                    mname, device=self.cfg.EMB_DEVICE)

                emb = model.encode(
                    clean.tolist(),
                    show_progress_bar=True,
                    batch_size=self.cfg.EMB_BATCH_SIZE,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

                elapsed = time.time() - t_start
                print(f"    Encoded {len(clean)} texts in {elapsed:.1f}s "
                      f"({len(clean)/elapsed:.0f} texts/sec)")

                tag = mname.replace("/", "_").replace("-", "_")
                if self.cfg.USE_PCA:
                    nc = min(self.cfg.PCA_N, *emb.shape)
                    pca = PCA(n_components=nc)
                    emb = pca.fit_transform(emb)
                    self.pcas[mname] = pca
                    evr = pca.explained_variance_ratio_.sum()
                    print(f"    PCA {nc} comp → {evr:.2%} var")
                    self._scree(pca, tag)
                    cols = [f"{tag}_pca_{i}" for i in range(nc)]
                else:
                    cols = [f"{tag}_{i}" for i in range(emb.shape[1])]
                out[tag] = pd.DataFrame(emb, columns=cols)

                if self.cfg.USE_GPU:
                    del model
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    SKIP: {e}")
        return out

    def tfidf(self, text_series):
        print("  TF-IDF …")
        clean = text_series.astype(str).fillna("")
        vec = TfidfVectorizer(
            max_features=self.cfg.TFIDF_MAX,
            stop_words="english", ngram_range=(1, 2))
        mat = vec.fit_transform(clean)
        nc = min(self.cfg.PCA_N, *mat.shape)
        pca = PCA(n_components=nc)
        red = pca.fit_transform(mat.toarray())
        cols = [f"tfidf_pca_{i}" for i in range(nc)]
        print(f"    → {nc} components")
        return pd.DataFrame(red, columns=cols)

    def cluster_plot(self, edf, labels, tag):
        if edf.shape[1] < 2:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(edf.iloc[:, 0], edf.iloc[:, 1],
                        c=labels, cmap="RdYlGn_r", alpha=.45, s=15)
        plt.colorbar(sc, ax=ax, label="Churn")
        ax.set_title(f"Embedding Clusters — {tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 f"emb_cluster_{tag}.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

    def _scree(self, pca, tag):
        cum = np.cumsum(pca.explained_variance_ratio_)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, len(cum) + 1),
               pca.explained_variance_ratio_,
               alpha=.7, label="Individual")
        ax.plot(range(1, len(cum) + 1), cum, "ro-",
                label="Cumulative")
        ax.axhline(.95, ls="--", c="k", alpha=.4)
        ax.set_xlabel("Component")
        ax.set_ylabel("Var Ratio")
        ax.set_title(f"Scree — {tag}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 f"scree_{tag}.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()


# ====================================================================
# SECTION 6 — MODEL TRAINER  ★ MASSIVELY FASTER TUNING ★
# ====================================================================
class Trainer:
    METRICS = ["Acc", "Prec", "Rec", "F1", "AUC", "PR_AUC", "MCC"]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.best_params = {}
        self.cv = {}
        self.final_model = None
        self.final_name = None
        self.final_scaler = None
        self._tuning_times = {}  # track per-model timing

    # ----------------------------------------------------------------
    # Model factory (GPU-aware)
    # ----------------------------------------------------------------
    def _models(self):
        xgb_params = {
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0,
            "random_state": self.cfg.SEED,
            "n_jobs": -1,
        }
        xgb_params.update(self.cfg.xgb_gpu_params())

        m = {
            "LR": LogisticRegression(
                max_iter=2000, random_state=self.cfg.SEED),
            "RF": RandomForestClassifier(
                random_state=self.cfg.SEED, n_jobs=-1),
            "XGB": xgb.XGBClassifier(**xgb_params),
            "MLP": MLPClassifier(
                random_state=self.cfg.SEED, early_stopping=True),
        }

        if HAS_LGBM:
            lgbm_params = {
                "random_state": self.cfg.SEED,
                "verbose": -1,
                "n_jobs": -1,
            }
            lgbm_params.update(self.cfg.lgbm_gpu_params())
            m["LGBM"] = lgb.LGBMClassifier(**lgbm_params)

        # Stacking ensemble
        stack_xgb_params = {
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "scale_pos_weight": 5,
            "n_estimators": 200,
            "max_depth": 7,
            "verbosity": 0,
            "random_state": self.cfg.SEED,
        }
        stack_xgb_params.update(self.cfg.xgb_gpu_params())

        stack_estimators = [
            ("lr", LogisticRegression(
                max_iter=2000, C=1, class_weight="balanced",
                random_state=self.cfg.SEED)),
            ("xgb", xgb.XGBClassifier(**stack_xgb_params)),
        ]
        if HAS_LGBM:
            stack_lgbm_params = {
                "is_unbalance": True,
                "n_estimators": 200,
                "verbose": -1,
                "random_state": self.cfg.SEED,
            }
            stack_lgbm_params.update(self.cfg.lgbm_gpu_params())
            stack_estimators.append(
                ("lgbm", lgb.LGBMClassifier(**stack_lgbm_params)))

        m["Stack"] = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(
                max_iter=2000, random_state=self.cfg.SEED),
            cv=3, n_jobs=-1, passthrough=False,
        )
        return m

    # ================================================================
    # ★ UNIFIED TUNING DISPATCHER
    # ================================================================
    def tune(self, X, y):
        strategy = self.cfg.TUNING_STRATEGY
        print(f"\n{'='*65}")
        print(f"HYPERPARAMETER TUNING — Strategy: {strategy.upper()}")
        print(f"{'='*65}")
        print(f"  Inner CV folds : {self.cfg.TUNE_CV_FOLDS}")
        print(f"  Models to tune : {self.cfg.TUNE_MODELS}")

        col_names = (list(X.columns)
                     if isinstance(X, pd.DataFrame) else None)
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        if self.cfg.USE_SMOTE:
            sm = SMOTE(random_state=self.cfg.SEED)
            Xr, yr = sm.fit_resample(Xs, y)
        else:
            Xr, yr = Xs, y
        if col_names is not None:
            Xr = pd.DataFrame(Xr, columns=col_names)

        total_t0 = time.time()

        if strategy == "optuna":
            self._tune_optuna(Xr, yr, col_names)
        elif strategy == "halving":
            self._tune_halving(Xr, yr, col_names)
        elif strategy == "randomized":
            self._tune_randomized(Xr, yr, col_names)
        else:  # "grid"
            self._tune_grid(Xr, yr, col_names)

        total_elapsed = time.time() - total_t0
        print(f"\n  ✓ Total tuning time: {total_elapsed:.1f}s")
        print(f"  Per-model breakdown:")
        for name, t in self._tuning_times.items():
            print(f"    {name:>6s}: {t:.1f}s")

    # ================================================================
    # ★ STRATEGY 1: OPTUNA (Bayesian + pruning) — FASTEST
    # ================================================================
    def _tune_optuna(self, X, y, col_names):
        """
        Optuna uses Tree-structured Parzen Estimator (TPE) to
        explore the space intelligently, plus MedianPruner to
        kill unpromising trials early during CV.

        Why it's faster:
        - Doesn't try every combination (Bayesian sampling)
        - Prunes bad trials after 1-2 CV folds instead of all 3
        - GPU-accelerated XGBoost/LGBM training within each trial
        """
        if not HAS_OPTUNA:
            print("  optuna not installed; falling back to randomized")
            return self._tune_randomized(X, y, col_names)

        Xv = X.values if isinstance(X, pd.DataFrame) else X
        yv = y.values if isinstance(y, pd.Series) else y
        inner_cv = StratifiedKFold(
            self.cfg.TUNE_CV_FOLDS, shuffle=True,
            random_state=self.cfg.SEED)

        for name in self.cfg.TUNE_MODELS:
            if name not in self.cfg.GRIDS:
                continue
            t0 = time.time()
            print(f"  {name} (Optuna, {self.cfg.OPTUNA_N_TRIALS} trials) …",
                  end=" ", flush=True)

            def objective(trial, model_name=name):
                params = self._optuna_suggest(trial, model_name)
                mdl = self._make_model_with_params(model_name, params)

                # ★ XGBoost-specific: use pruning callback for
                #   early stopping of bad trials mid-CV
                scores = []
                for fold_i, (tr_i, vl_i) in enumerate(
                        inner_cv.split(Xv, yv)):
                    mc = clone(mdl)
                    mc.fit(Xv[tr_i], yv[tr_i])
                    yp = mc.predict(Xv[vl_i])
                    score = f1_score(yv[vl_i], yp, zero_division=0)
                    scores.append(score)

                    # Report intermediate score for pruning
                    trial.report(np.mean(scores), fold_i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return np.mean(scores)

            # Create study with pruner
            pruner = (optuna.pruners.MedianPruner(n_warmup_steps=1)
                      if self.cfg.OPTUNA_PRUNING
                      else optuna.pruners.NopPruner())

            study = optuna.create_study(
                direction="maximize",
                pruner=pruner,
                sampler=optuna.samplers.TPESampler(
                    seed=self.cfg.SEED),
            )
            study.optimize(
                objective,
                n_trials=self.cfg.OPTUNA_N_TRIALS,
                timeout=self.cfg.OPTUNA_TIMEOUT,
                show_progress_bar=False,
            )

            elapsed = time.time() - t0
            self._tuning_times[name] = elapsed
            self.best_params[name] = study.best_params
            n_pruned = len([t for t in study.trials
                           if t.state == optuna.trial.TrialState.PRUNED])
            print(f"F1={study.best_value:.4f}  "
                  f"({elapsed:.1f}s, {n_pruned} pruned)  "
                  f"{study.best_params}")

    def _optuna_suggest(self, trial, name):
        """Map Optuna trial suggestions to parameter grids."""
        grid = self.cfg.GRIDS[name]
        params = {}
        for key, values in grid.items():
            if all(isinstance(v, (int, float)) for v in values):
                if all(isinstance(v, int) for v in values):
                    params[key] = trial.suggest_int(
                        key, min(values), max(values))
                else:
                    # For learning rates etc, use log scale
                    mn, mx = min(values), max(values)
                    if mn > 0 and mx / mn > 10:
                        params[key] = trial.suggest_float(
                            key, mn, mx, log=True)
                    else:
                        params[key] = trial.suggest_float(
                            key, mn, mx)
            elif all(isinstance(v, bool) for v in values):
                params[key] = trial.suggest_categorical(key, values)
            elif all(isinstance(v, tuple) for v in values):
                params[key] = trial.suggest_categorical(key, values)
            elif all(isinstance(v, str) or v is None
                     for v in values):
                params[key] = trial.suggest_categorical(key, values)
            else:
                params[key] = trial.suggest_categorical(key, values)
        return params

    def _make_model_with_params(self, name, params):
        """Create a fresh model instance with given params."""
        mdl = self._models()[name]
        # Filter params to only those the model accepts
        valid = mdl.get_params().keys()
        filtered = {k: v for k, v in params.items() if k in valid}
        mdl.set_params(**filtered)
        return mdl

    # ================================================================
    # ★ STRATEGY 2: HALVING SEARCH — fast, no extra dependencies
    # ================================================================
    def _tune_halving(self, X, y, col_names):
        """
        HalvingRandomSearchCV starts with many candidates on a
        small subset of data, then progressively increases data
        while eliminating poor performers.

        Why it's faster:
        - Most candidates only see 1/8 of the data
        - Only survivors see the full dataset
        - Built into sklearn (no extra package needed)
        """
        if not HAS_HALVING:
            print("  HalvingRandomSearchCV not available; "
                  "falling back to randomized")
            return self._tune_randomized(X, y, col_names)

        inner_cv = StratifiedKFold(
            self.cfg.TUNE_CV_FOLDS, shuffle=True,
            random_state=self.cfg.SEED)

        for name, mdl in self._models().items():
            if name not in self.cfg.GRIDS or name not in self.cfg.TUNE_MODELS:
                continue
            t0 = time.time()
            grid_size = self._grid_size(self.cfg.GRIDS[name])
            print(f"  {name} (Halving, {grid_size} candidates) …",
                  end=" ", flush=True)

            search = HalvingRandomSearchCV(
                mdl, self.cfg.GRIDS[name],
                n_candidates=min(grid_size, 60),
                cv=inner_cv,
                scoring="f1",
                factor=3,          # eliminate 2/3 each round
                min_resources=50,  # min samples per candidate
                random_state=self.cfg.SEED,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X, y)

            elapsed = time.time() - t0
            self._tuning_times[name] = elapsed
            self.best_params[name] = search.best_params_
            print(f"F1={search.best_score_:.4f}  "
                  f"({elapsed:.1f}s)  {search.best_params_}")

    # ================================================================
    # ★ STRATEGY 3: RANDOMIZED SEARCH — good baseline
    # ================================================================
    def _tune_randomized(self, X, y, col_names):
        inner_cv = StratifiedKFold(
            self.cfg.TUNE_CV_FOLDS, shuffle=True,
            random_state=self.cfg.SEED)

        for name, mdl in self._models().items():
            if name not in self.cfg.GRIDS or name not in self.cfg.TUNE_MODELS:
                continue
            t0 = time.time()
            grid_size = self._grid_size(self.cfg.GRIDS[name])
            n_iter = min(self.cfg.N_ITER_RANDOM, grid_size)
            print(f"  {name} (Randomized, {n_iter}/{grid_size}) …",
                  end=" ", flush=True)

            gs = RandomizedSearchCV(
                mdl, self.cfg.GRIDS[name],
                n_iter=n_iter,
                cv=inner_cv,
                scoring="f1",
                n_jobs=-1,
                verbose=0,
                random_state=self.cfg.SEED,
            )
            gs.fit(X, y)

            elapsed = time.time() - t0
            self._tuning_times[name] = elapsed
            self.best_params[name] = gs.best_params_
            print(f"F1={gs.best_score_:.4f}  "
                  f"({elapsed:.1f}s)  {gs.best_params_}")

    # ================================================================
    # STRATEGY 4: GRID SEARCH — exhaustive (slowest)
    # ================================================================
    def _tune_grid(self, X, y, col_names):
        inner_cv = StratifiedKFold(
            self.cfg.TUNE_CV_FOLDS, shuffle=True,
            random_state=self.cfg.SEED)

        for name, mdl in self._models().items():
            if name not in self.cfg.GRIDS or name not in self.cfg.TUNE_MODELS:
                continue
            t0 = time.time()
            grid_size = self._grid_size(self.cfg.GRIDS[name])
            print(f"  {name} (Grid, {grid_size} combos) …",
                  end=" ", flush=True)

            gs = GridSearchCV(
                mdl, self.cfg.GRIDS[name],
                cv=inner_cv,
                scoring="f1",
                n_jobs=-1,
                verbose=0,
            )
            gs.fit(X, y)

            elapsed = time.time() - t0
            self._tuning_times[name] = elapsed
            self.best_params[name] = gs.best_params_
            print(f"F1={gs.best_score_:.4f}  "
                  f"({elapsed:.1f}s)  {gs.best_params_}")

    @staticmethod
    def _grid_size(grid):
        size = 1
        for v in grid.values():
            size *= len(v)
        return size

    # ================================================================
    # CROSS-VALIDATION (unchanged)
    # ================================================================
    def evaluate_cv(self, X, y, tag="exp"):
        print(f"\n--- {self.cfg.N_SPLITS}-Fold CV : {tag} ---")
        skf = StratifiedKFold(
            self.cfg.N_SPLITS, shuffle=True,
            random_state=self.cfg.SEED)
        models = self._models()
        for nm in models:
            if nm in self.best_params:
                models[nm].set_params(**self.best_params[nm])

        raw = {n: {m: [] for m in self.METRICS} for n in models}
        pred = {n: {"yt": [], "yp": [], "yb": []} for n in models}

        col_names = (list(X.columns)
                     if isinstance(X, pd.DataFrame) else None)
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        yv = y.values if isinstance(y, pd.Series) else y

        for fold_i, (train_i, val_i) in enumerate(skf.split(Xv, yv)):
            Xtr, Xvl = Xv[train_i], Xv[val_i]
            ytr, yvl = yv[train_i], yv[val_i]
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xvl = sc.transform(Xvl)
            if self.cfg.USE_SMOTE:
                Xtr, ytr = SMOTE(
                    random_state=self.cfg.SEED).fit_resample(Xtr, ytr)
            if col_names is not None:
                Xtr = pd.DataFrame(Xtr, columns=col_names)
                Xvl = pd.DataFrame(Xvl, columns=col_names)
            for nm, mdl in models.items():
                mc = clone(mdl)
                mc.fit(Xtr, ytr)
                yp = mc.predict(Xvl)
                yb = mc.predict_proba(Xvl)[:, 1]
                raw[nm]["Acc"].append(accuracy_score(yvl, yp))
                raw[nm]["Prec"].append(
                    precision_score(yvl, yp, zero_division=0))
                raw[nm]["Rec"].append(
                    recall_score(yvl, yp, zero_division=0))
                raw[nm]["F1"].append(
                    f1_score(yvl, yp, zero_division=0))
                raw[nm]["AUC"].append(roc_auc_score(yvl, yb))
                raw[nm]["PR_AUC"].append(
                    average_precision_score(yvl, yb))
                raw[nm]["MCC"].append(matthews_corrcoef(yvl, yp))
                pred[nm]["yt"].extend(yvl.tolist())
                pred[nm]["yp"].extend(yp.tolist())
                pred[nm]["yb"].extend(yb.tolist())

        summary = {}
        for nm in models:
            summary[nm] = {}
            for m in self.METRICS:
                summary[nm][f"{m}_mean"] = np.mean(raw[nm][m])
                summary[nm][f"{m}_std"] = np.std(raw[nm][m])

        hdr = (f"{'Model':<8}"
               + "".join(f"{'  '+m:>13}"
                         for m in ["F1", "AUC", "PR_AUC", "MCC"]))
        print(hdr)
        print("-" * len(hdr))
        for nm in models:
            s = summary[nm]
            vals = "  ".join(
                f"{s[f'{m}_mean']:.4f}±{s[f'{m}_std']:.4f}"
                for m in ["F1", "AUC", "PR_AUC", "MCC"])
            print(f"{nm:<8}  {vals}")

        self.cv[tag] = {
            "summary": summary, "raw": raw, "preds": pred}
        return summary, raw, pred

    def train_final(self, Xtr, ytr, Xte, yte, name=None):
        print("\n--- Training Final Model ---")
        if name is None:
            best_f1, name = 0, "XGB"
            for _, d in self.cv.items():
                for nm, s in d["summary"].items():
                    if s["F1_mean"] > best_f1:
                        best_f1, name = s["F1_mean"], nm
        mdl = self._models()[name]
        if name in self.best_params:
            mdl.set_params(**self.best_params[name])

        col_names = (list(Xtr.columns)
                     if isinstance(Xtr, pd.DataFrame) else None)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        if self.cfg.USE_SMOTE:
            Xtr_s, ytr = SMOTE(
                random_state=self.cfg.SEED).fit_resample(Xtr_s, ytr)
        if col_names is not None:
            Xtr_s = pd.DataFrame(Xtr_s, columns=col_names)
            Xte_s = pd.DataFrame(Xte_s, columns=col_names)
        mdl.fit(Xtr_s, ytr)
        yp = mdl.predict(Xte_s)
        yb = mdl.predict_proba(Xte_s)[:, 1]

        self.final_model = mdl
        self.final_name = name
        self.final_scaler = sc

        print(f"\nBest model: {name}")
        print(classification_report(
            yte, yp, target_names=["No Churn", "Churn"]))
        return mdl, sc, Xtr_s, Xte_s, yp, yb


# ====================================================================
# SECTION 7 — EVALUATION / VISUALISATION ENGINE
# ====================================================================
class Evaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def ablation_chart(self, cv_store):
        rows = []
        for exp, d in cv_store.items():
            for nm, s in d["summary"].items():
                rows.append({"Experiment": exp, "Model": nm, **s})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.cfg.RESULT_DIR, "ablation.csv"),
                  index=False)
        for met in ["F1", "AUC", "PR_AUC", "MCC"]:
            fig, ax = plt.subplots(figsize=(12, 5))
            exps = df["Experiment"].unique()
            mdls = df["Model"].unique()
            x = np.arange(len(exps))
            w = .8 / len(mdls)
            for i, m in enumerate(mdls):
                sub = df[df["Model"] == m].set_index("Experiment")
                mn = [sub.loc[e, f"{met}_mean"]
                      if e in sub.index else 0 for e in exps]
                sd = [sub.loc[e, f"{met}_std"]
                      if e in sub.index else 0 for e in exps]
                ax.bar(x + i * w, mn, w, yerr=sd,
                       label=m, capsize=2)
            ax.set_xticks(x + w * (len(mdls) - 1) / 2)
            ax.set_xticklabels(exps, rotation=40, ha="right",
                               fontsize=7)
            ax.set_ylabel(met)
            ax.set_title(f"Ablation — {met}")
            ax.legend(fontsize=7)
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                     f"ablation_{met}.png"),
                        dpi=self.cfg.FIG_DPI)
            plt.close()

    def roc(self, preds, tag=""):
        fig, ax = plt.subplots(figsize=(7, 5))
        for nm, p in preds.items():
            yt, yb = np.array(p["yt"]), np.array(p["yb"])
            fpr, tpr, _ = roc_curve(yt, yb)
            ax.plot(fpr, tpr,
                    label=f"{nm} ({roc_auc_score(yt, yb):.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC — {tag}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 f"roc_{tag}.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

    def pr(self, preds, tag=""):
        fig, ax = plt.subplots(figsize=(7, 5))
        for nm, p in preds.items():
            yt, yb = np.array(p["yt"]), np.array(p["yb"])
            prec, rec, _ = precision_recall_curve(yt, yb)
            ax.plot(rec, prec,
                    label=f"{nm} "
                          f"({average_precision_score(yt, yb):.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {tag}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 f"pr_{tag}.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

    def cm(self, preds, tag=""):
        n = len(preds)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
        if n == 1:
            axes = [axes]
        for i, (nm, p) in enumerate(preds.items()):
            c = confusion_matrix(p["yt"], p["yp"])
            sns.heatmap(c, annot=True, fmt="d", cmap="Blues",
                        ax=axes[i],
                        xticklabels=["No", "Yes"],
                        yticklabels=["No", "Yes"])
            axes[i].set_title(nm)
            axes[i].set_ylabel("True")
            axes[i].set_xlabel("Pred")
        plt.suptitle(f"Confusion Matrices — {tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 f"cm_{tag}.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

    def calibration(self, preds, tag=""):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        for nm, p in preds.items():
            try:
                pt, pp = calibration_curve(p["yt"], p["yb"],
                                           n_bins=10)
                ax.plot(pp, pt, "s-", label=nm)
            except Exception:
                pass
        ax.set_xlabel("Mean Predicted Prob")
        ax.set_ylabel("Fraction +")
        ax.set_title(f"Calibration — {tag}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 f"calib_{tag}.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

    def learn_curve(self, mdl, X, y, name):
        print(f"  Learning curve — {name}")
        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            ts, tr, vl = learning_curve(
                mdl, X, y, cv=3, scoring="f1",
                train_sizes=np.linspace(.1, 1., 8),
                n_jobs=-1, random_state=self.cfg.SEED)
            ax.fill_between(ts, tr.mean(1) - tr.std(1),
                            tr.mean(1) + tr.std(1),
                            alpha=.1, color="b")
            ax.fill_between(ts, vl.mean(1) - vl.std(1),
                            vl.mean(1) + vl.std(1),
                            alpha=.1, color="orange")
            ax.plot(ts, tr.mean(1), "o-", c="b", label="Train")
            ax.plot(ts, vl.mean(1), "o-", c="orange", label="Val")
            ax.set_xlabel("Train Size")
            ax.set_ylabel("F1")
            ax.set_title(f"Learning Curve — {name}")
            ax.legend()
        except Exception as e:
            ax.text(.5, .5, str(e), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 f"lc_{name}.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

    def opt_threshold(self, yt, yb):
        ths = np.arange(.1, .9, .01)
        f1s = [f1_score(yt, (yb >= t).astype(int), zero_division=0)
               for t in ths]
        bi = int(np.argmax(f1s))
        bt, bf = ths[bi], f1s[bi]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ths, f1s, "b-")
        ax.axvline(bt, ls="--", c="r",
                   label=f"Best {bt:.2f} (F1={bf:.4f})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1")
        ax.set_title("Threshold Optimisation")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR, "threshold.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()
        print(f"  Optimal threshold: {bt:.2f}  (F1={bf:.4f})")
        return bt


# ====================================================================
# SECTION 8 — STATISTICAL TESTING
# ====================================================================
class StatTests:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, raw, preds):
        print("\n" + "=" * 65)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 65)
        res = {
            "friedman": self._friedman(raw),
            "mcnemar": [],
            "ttest": [],
        }
        names = list(preds.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                res["mcnemar"].append(
                    self._mcnemar(preds[a], preds[b], a, b))
                res["ttest"].append(
                    self._paired_t(raw[a], raw[b], a, b))

        df = pd.DataFrame(res["mcnemar"])
        if not df.empty:
            df.to_csv(os.path.join(self.cfg.RESULT_DIR,
                                   "mcnemar.csv"), index=False)
        df2 = pd.DataFrame(res["ttest"])
        if not df2.empty:
            df2.to_csv(os.path.join(self.cfg.RESULT_DIR,
                                    "ttest.csv"), index=False)
        return res

    def _friedman(self, raw):
        names = list(raw.keys())
        if len(names) < 3:
            print("  Friedman: need ≥3 models, skipped.")
            return None
        arrays = [raw[n]["F1"] for n in names]
        min_len = min(len(a) for a in arrays)
        arrays = [a[:min_len] for a in arrays]
        try:
            stat, p = stats.friedmanchisquare(*arrays)
            print(f"  Friedman χ²={stat:.4f}  p={p:.6f}"
                  f"  {'SIG' if p < .05 else 'n.s.'}")
            if p < 0.05 and len(names) >= 3:
                self._nemenyi(arrays, names)
            return {"stat": stat, "p": p}
        except Exception as e:
            print(f"  Friedman failed: {e}")
            return None

    def _nemenyi(self, arrays, names):
        print("  Nemenyi post-hoc "
              "(pairwise Wilcoxon with Bonferroni):")
        from itertools import combinations
        n_comp = len(list(combinations(range(len(names)), 2)))
        for i, j in combinations(range(len(names)), 2):
            try:
                stat, p = stats.wilcoxon(arrays[i], arrays[j])
                adj_p = min(p * n_comp, 1.0)
                sig = "SIG" if adj_p < .05 else "n.s."
                print(f"    {names[i]} vs {names[j]}: "
                      f"p_adj={adj_p:.6f} {sig}")
            except Exception:
                print(f"    {names[i]} vs {names[j]}: "
                      f"insufficient data")

    def _mcnemar(self, pa, pb, na, nb):
        ya = np.array(pa["yt"])
        a_correct = (np.array(pa["yp"]) == ya).astype(int)
        b_correct = (np.array(pb["yp"]) == ya).astype(int)
        n01 = int(np.sum((a_correct == 1) & (b_correct == 0)))
        n10 = int(np.sum((a_correct == 0) & (b_correct == 1)))
        if n01 + n10 == 0:
            print(f"  McNemar {na} vs {nb}: identical predictions")
            return {"A": na, "B": nb, "chi2": 0, "p": 1.0,
                    "sig": "n.s."}
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p = 1 - stats.chi2.cdf(chi2, 1)
        sig = "SIG" if p < .05 else "n.s."
        print(f"  McNemar {na} vs {nb}: "
              f"χ²={chi2:.4f} p={p:.6f} {sig}")
        return {"A": na, "B": nb, "chi2": chi2, "p": p, "sig": sig}

    def _paired_t(self, ra, rb, na, nb):
        a_f1 = ra["F1"]
        b_f1 = rb["F1"]
        min_len = min(len(a_f1), len(b_f1))
        if min_len < 2:
            return {"A": na, "B": nb, "t": 0, "p": 1.0,
                    "sig": "n.s."}
        t, p = stats.ttest_rel(a_f1[:min_len], b_f1[:min_len])
        sig = "SIG" if p < .05 else "n.s."
        print(f"  Paired-t {na} vs {nb}: "
              f"t={t:.4f} p={p:.6f} {sig}")
        return {"A": na, "B": nb, "t": t, "p": p, "sig": sig}

    def bootstrap_ci(self, yt, yb, metric_fn,
                     n_boot=1000, alpha=.05):
        rng = np.random.RandomState(self.cfg.SEED)
        scores = []
        yt, yb = np.array(yt), np.array(yb)
        for _ in range(n_boot):
            idx = rng.choice(len(yt), len(yt), replace=True)
            try:
                scores.append(metric_fn(yt[idx], yb[idx]))
            except Exception:
                pass
        lo = np.percentile(scores, 100 * alpha / 2)
        hi = np.percentile(scores, 100 * (1 - alpha / 2))
        mn = np.mean(scores)
        print(f"  Bootstrap ({n_boot}): "
              f"{mn:.4f} [{lo:.4f}, {hi:.4f}]")
        return {"mean": mn, "lo": lo, "hi": hi}


# ====================================================================
# SECTION 9 — EXPLAINABILITY ENGINE (SHAP + LIME)
# ====================================================================
class Explainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def shap_global(self, model, X, feature_names,
                    model_name="model"):
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names")
        print("\n" + "=" * 65)
        print(f"SHAP GLOBAL ANALYSIS — {model_name}")
        print("=" * 65)

        Xv = X if isinstance(X, np.ndarray) else X.values
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()

        tree_types = (xgb.XGBClassifier, RandomForestClassifier)
        if HAS_LGBM:
            tree_types = tree_types + (lgb.LGBMClassifier,)

        if isinstance(model, tree_types):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, Xv)
        else:
            bg = shap.sample(Xv, min(50, len(Xv)))
            explainer = shap.KernelExplainer(model.predict_proba, bg)

        sv = explainer.shap_values(Xv)
        if isinstance(sv, list):
            sv = sv[1]
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(sv, Xv, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance — {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 f"shap_bar_{model_name}.png"),
                    dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close("all")

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(sv, Xv, feature_names=feature_names,
                          show=False, max_display=20)
        plt.title(f"SHAP Beeswarm — {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 f"shap_bees_{model_name}.png"),
                    dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close("all")

        mean_abs = np.abs(sv).mean(0)
        top5 = np.argsort(mean_abs)[-5:][::-1]
        for rank, idx in enumerate(top5):
            fig, ax = plt.subplots(figsize=(6, 4))
            shap.dependence_plot(idx, sv, Xv,
                                 feature_names=feature_names,
                                 show=False, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(
                self.cfg.FIG_DIR,
                f"shap_dep_{model_name}_{rank}_"
                f"{feature_names[idx]}.png"),
                dpi=self.cfg.FIG_DPI, bbox_inches="tight")
            plt.close("all")

        self._modality_split(sv, feature_names)
        return sv

    def _modality_split(self, sv, names):
        text_kw = ("pca_", "tfidf_", "emb_")
        txt_idx = [i for i, n in enumerate(names)
                   if any(k in n for k in text_kw)]
        str_idx = [i for i in range(len(names))
                   if i not in txt_idx]

        if not txt_idx:
            print("  No text-derived features found; "
                  "skipping modality split.")
            return

        txt_imp = np.abs(sv[:, txt_idx]).sum()
        str_imp = np.abs(sv[:, str_idx]).sum()
        total = txt_imp + str_imp
        if total == 0:
            return
        print(f"  Structured importance : {str_imp/total:.2%}")
        print(f"  Text-derived import.  : {txt_imp/total:.2%}")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Structured", "Text-derived"],
               [str_imp / total, txt_imp / total],
               color=["#3498db", "#e67e22"])
        ax.set_ylabel("Share of |SHAP|")
        ax.set_title("Modality Contribution")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 "modality_split.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

    def shap_local(self, model, X, feature_names, indices=None,
                   model_name="model"):
        print(f"\n--- SHAP Local Explanations — {model_name} ---")
        Xv = X if isinstance(X, np.ndarray) else X.values
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()

        tree_types = (xgb.XGBClassifier, RandomForestClassifier)
        if HAS_LGBM:
            tree_types = tree_types + (lgb.LGBMClassifier,)

        if isinstance(model, tree_types):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, Xv)
        else:
            bg = shap.sample(Xv, min(50, len(Xv)))
            explainer = shap.KernelExplainer(model.predict_proba, bg)

        if indices is None:
            indices = list(range(min(3, len(Xv))))

        for ci in indices:
            row = Xv[ci:ci + 1]
            sv = explainer.shap_values(row)
            if isinstance(sv, list):
                sv = sv[1]
            if sv.ndim == 3:
                sv = sv[:, :, 1]

            vals = sv.flatten()
            top_n = min(10, len(vals))
            order = np.argsort(np.abs(vals))[-top_n:][::-1]
            print(f"\n  Customer {ci}:")
            for r, idx in enumerate(order):
                print(f"    {feature_names[idx]:>25s}  "
                      f"SHAP={vals[idx]:+.4f}  "
                      f"val={Xv[ci, idx]:.4f}")

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#e74c3c" if v > 0 else "#2ecc71"
                      for v in vals[order]]
            ax.barh([feature_names[i] for i in order],
                    vals[order], color=colors)
            ax.set_xlabel("SHAP Value")
            ax.set_title(f"Local Explanation — Customer {ci}")
            plt.tight_layout()
            plt.savefig(os.path.join(
                self.cfg.FIG_DIR,
                f"shap_local_{model_name}_{ci}.png"),
                dpi=self.cfg.FIG_DPI)
            plt.close()

    def lime_compare(self, model, X, feature_names, idx=0,
                     model_name="model"):
        if not HAS_LIME:
            print("  lime not installed; skipping.")
            return None
        print(f"\n--- LIME vs SHAP Comparison — "
              f"Customer {idx} ---")
        Xv = X if isinstance(X, np.ndarray) else X.values
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()

        explr = lime.lime_tabular.LimeTabularExplainer(
            Xv, feature_names=feature_names,
            class_names=["No Churn", "Churn"],
            mode="classification",
            random_state=self.cfg.SEED)
        exp = explr.explain_instance(
            Xv[idx], model.predict_proba, num_features=10)
        lime_top = [f for f, _ in exp.as_list()]

        tree_types = (xgb.XGBClassifier, RandomForestClassifier)
        if HAS_LGBM:
            tree_types = tree_types + (lgb.LGBMClassifier,)
        if isinstance(model, tree_types):
            expl = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            expl = shap.LinearExplainer(model, Xv)
        else:
            bg = shap.sample(Xv, min(50, len(Xv)))
            expl = shap.KernelExplainer(model.predict_proba, bg)

        sv = expl.shap_values(Xv[idx:idx + 1])
        if isinstance(sv, list):
            sv = sv[1]
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        sv = sv.flatten()
        shap_ord = np.argsort(np.abs(sv))[-10:][::-1]
        shap_top = [feature_names[i] for i in shap_ord]

        def _clean(s):
            for c in "<>=!":
                s = s.split(c)[0].strip()
            return s
        lime_clean = [_clean(f) for f in lime_top]
        overlap = set(lime_clean) & set(shap_top)
        print(f"  LIME top-10: {lime_clean}")
        print(f"  SHAP top-10: {shap_top}")
        print(f"  Overlap    : {len(overlap)} / 10  → {overlap}")

        try:
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(8, 5)
            plt.title(f"LIME — Customer {idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(
                self.cfg.FIG_DIR,
                f"lime_{model_name}_{idx}.png"),
                dpi=self.cfg.FIG_DPI, bbox_inches="tight")
            plt.close("all")
        except Exception:
            pass

        return {"overlap": len(overlap),
                "lime_top": lime_clean,
                "shap_top": shap_top}

    def permutation_importance(self, model, X, y, feature_names,
                               model_name="model"):
        from sklearn.inspection import (
            permutation_importance as perm_imp)
        print(f"\n--- Permutation Importance — {model_name} ---")
        Xv = X
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()
        result = perm_imp(
            model, Xv, y, n_repeats=10,
            random_state=self.cfg.SEED,
            scoring="f1", n_jobs=-1)
        order = result.importances_mean.argsort()[-15:][::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            [feature_names[i] for i in reversed(order)],
            result.importances_mean[list(reversed(order))],
            xerr=result.importances_std[list(reversed(order))],
            capsize=2, color="#3498db")
        ax.set_xlabel("ΔF1")
        ax.set_title(f"Permutation Importance — {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 f"perm_imp_{model_name}.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()
        return result


# ====================================================================
# SECTION 10 — PRESCRIPTIVE RETENTION OPTIMIZER
# ====================================================================
class Optimizer:
    ACTIONS = {
        "no_action":       {"cost": 0,   "success": 0.00},
        "10pct_discount":  {"cost": 50,  "success": 0.20},
        "25pct_discount":  {"cost": 120, "success": 0.40},
        "premium_support": {"cost": 80,  "success": 0.30},
        "loyalty_reward":  {"cost": 60,  "success": 0.25},
        "personal_call":   {"cost": 30,  "success": 0.15},
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def optimise(self, cust_df, budget=None):
        print("\n" + "=" * 65)
        print("RETENTION OPTIMISATION (LP)")
        print("=" * 65)
        if budget is None:
            budget = self.cfg.BUDGET

        cust = cust_df.copy()
        cust["clv"] = cust["monthly_charges"] * 12 * 1.2

        actions = {k: v for k, v in self.ACTIONS.items()
                   if k != "no_action"}
        prob = LpProblem("Retention", LpMaximize)
        x = {}
        for _, row in cust.iterrows():
            ci = int(row["customer_idx"])
            for a in actions:
                x[(ci, a)] = LpVariable(f"x_{ci}_{a}", cat="Binary")

        prob += lpSum(
            x[(int(r["customer_idx"]), a)]
            * r["churn_prob"]
            * actions[a]["success"]
            * r["clv"]
            for _, r in cust.iterrows()
            for a in actions
        )

        prob += lpSum(
            x[(int(r["customer_idx"]), a)] * actions[a]["cost"]
            for _, r in cust.iterrows()
            for a in actions
        ) <= budget

        for _, r in cust.iterrows():
            ci = int(r["customer_idx"])
            prob += lpSum(x[(ci, a)] for a in actions) <= 1

        prob.solve(PULP_CBC_CMD(msg=0))

        rows = []
        for _, r in cust.iterrows():
            ci = int(r["customer_idx"])
            chosen = "no_action"
            for a in actions:
                if value(x[(ci, a)]) == 1:
                    chosen = a
                    break
            info = self.ACTIONS[chosen]
            rows.append({
                "customer_idx": ci,
                "churn_prob": r["churn_prob"],
                "monthly_charges": r["monthly_charges"],
                "clv": r["clv"],
                "action": chosen,
                "cost": info["cost"],
                "success_rate": info["success"],
                "expected_saved": (r["churn_prob"]
                                   * info["success"] * r["clv"]),
            })
        res = pd.DataFrame(rows)

        targeted = res[res["action"] != "no_action"]
        total_cost = targeted["cost"].sum()
        total_saved = targeted["expected_saved"].sum()
        roi = ((total_saved - total_cost) / total_cost * 100
               if total_cost > 0 else 0)

        print(f"  Budget        : ${budget:,.0f}")
        print(f"  Total cost    : ${total_cost:,.0f}")
        print(f"  Expected saved: ${total_saved:,.0f}")
        print(f"  ROI           : {roi:,.1f}%")
        print(f"  Customers targeted: "
              f"{len(targeted)} / {len(res)}")
        print(f"  Action distribution:")
        for a, cnt in targeted["action"].value_counts().items():
            print(f"    {a:>20s}: {cnt}")

        res.to_csv(os.path.join(self.cfg.RESULT_DIR,
                                "retention_plan.csv"), index=False)
        return res

    def sensitivity(self, cust_df, budgets=None):
        print("\n--- Budget Sensitivity Analysis ---")
        if budgets is None:
            budgets = [1000, 3000, 5000, 10000, 15000]

        rows = []
        for b in budgets:
            res = self._quick_solve(cust_df, b)
            t = res[res["action"] != "no_action"]
            cost = t["cost"].sum()
            saved = t["expected_saved"].sum()
            roi = ((saved - cost) / cost * 100
                   if cost > 0 else 0)
            rows.append({"budget": b, "cost": cost,
                         "saved": saved, "roi": roi,
                         "n_targeted": len(t)})
            print(f"  Budget ${b:>7,}  → targeted {len(t):>4}  "
                  f"ROI {roi:>7.1f}%")

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.cfg.RESULT_DIR,
                               "sensitivity.csv"), index=False)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(df["budget"], df["roi"], "o-", c="#e74c3c")
        axes[0].set_xlabel("Budget ($)")
        axes[0].set_ylabel("ROI (%)")
        axes[0].set_title("ROI vs Budget")

        axes[1].plot(df["budget"], df["saved"], "s-", c="#3498db")
        axes[1].set_xlabel("Budget ($)")
        axes[1].set_ylabel("Expected Saved ($)")
        axes[1].set_title("Value Saved vs Budget")

        axes[2].plot(df["budget"], df["n_targeted"], "^-",
                     c="#2ecc71")
        axes[2].set_xlabel("Budget ($)")
        axes[2].set_ylabel("Customers Targeted")
        axes[2].set_title("Coverage vs Budget")

        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 "sensitivity.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()
        return df

    def _quick_solve(self, cust_df, budget):
        cust = cust_df.copy()
        if "clv" not in cust.columns:
            cust["clv"] = cust["monthly_charges"] * 12 * 1.2

        actions = {k: v for k, v in self.ACTIONS.items()
                   if k != "no_action"}
        prob = LpProblem("Ret_sens", LpMaximize)
        x = {}
        for _, r in cust.iterrows():
            ci = int(r["customer_idx"])
            for a in actions:
                x[(ci, a)] = LpVariable(f"x_{ci}_{a}",
                                        cat="Binary")

        prob += lpSum(
            x[(int(r["customer_idx"]), a)]
            * r["churn_prob"] * actions[a]["success"] * r["clv"]
            for _, r in cust.iterrows() for a in actions
        )
        prob += lpSum(
            x[(int(r["customer_idx"]), a)] * actions[a]["cost"]
            for _, r in cust.iterrows() for a in actions
        ) <= budget
        for _, r in cust.iterrows():
            ci = int(r["customer_idx"])
            prob += lpSum(x[(ci, a)] for a in actions) <= 1

        prob.solve(PULP_CBC_CMD(msg=0))

        rows = []
        for _, r in cust.iterrows():
            ci = int(r["customer_idx"])
            chosen = "no_action"
            for a in actions:
                if value(x[(ci, a)]) == 1:
                    chosen = a
                    break
            info = self.ACTIONS[chosen]
            rows.append({
                "customer_idx": ci,
                "churn_prob": r["churn_prob"],
                "monthly_charges": r["monthly_charges"],
                "clv": r["clv"],
                "action": chosen,
                "cost": info["cost"],
                "success_rate": info["success"],
                "expected_saved": (r["churn_prob"]
                                   * info["success"] * r["clv"]),
            })
        return pd.DataFrame(rows)

    def monte_carlo(self, opt_result, n_sim=2000):
        print(f"\n--- Monte Carlo Simulation ({n_sim} runs) ---")
        targeted = opt_result[
            opt_result["action"] != "no_action"].copy()
        if targeted.empty:
            print("  No customers targeted; skipping MC.")
            return None

        rng = np.random.RandomState(self.cfg.SEED)
        total_cost = targeted["cost"].sum()

        sr_base = targeted["success_rate"].values
        cp_base = targeted["churn_prob"].values
        clv = targeted["clv"].values
        n_cust = len(targeted)

        sr_mult = rng.uniform(0.7, 1.3, size=(n_sim, n_cust))
        cp_mult = rng.uniform(0.8, 1.2, size=(n_sim, n_cust))
        coin = rng.random(size=(n_sim, n_cust))

        sr_perturbed = np.clip(
            sr_base[None, :] * sr_mult, 0, 1)
        cp_perturbed = np.clip(
            cp_base[None, :] * cp_mult, 0, 1)
        retained = (coin < sr_perturbed).astype(float)
        sim_saved = (
            retained * cp_perturbed * clv[None, :]).sum(axis=1)
        net_values = sim_saved - total_cost

        net_values = np.array(net_values)
        mn = net_values.mean()
        sd = net_values.std()
        lo = np.percentile(net_values, 5)
        hi = np.percentile(net_values, 95)
        p_pos = (net_values > 0).mean()

        print(f"  Mean net value  : ${mn:,.0f}")
        print(f"  Std             : ${sd:,.0f}")
        print(f"  90% CI          : [${lo:,.0f}, ${hi:,.0f}]")
        print(f"  P(positive ROI) : {p_pos:.2%}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(net_values, bins=60, color="#3498db",
                edgecolor="white", alpha=.8)
        ax.axvline(mn, c="red", ls="--",
                   label=f"Mean ${mn:,.0f}")
        ax.axvline(lo, c="orange", ls=":",
                   label=f"5th pctl ${lo:,.0f}")
        ax.axvline(hi, c="orange", ls=":",
                   label=f"95th pctl ${hi:,.0f}")
        ax.axvline(0, c="black", lw=1.5)
        ax.set_xlabel("Net Value ($)")
        ax.set_ylabel("Frequency")
        ax.set_title("Monte Carlo — Net Retention Value")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 "monte_carlo.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

        return {"mean": mn, "std": sd, "ci_5": lo, "ci_95": hi,
                "p_positive": p_pos}

    def compare_strategies(self, cust_df, budget=None):
        print("\n--- Strategy Comparison: "
              "LP vs Greedy vs Random ---")
        if budget is None:
            budget = self.cfg.BUDGET
        cust = cust_df.copy()
        if "clv" not in cust.columns:
            cust["clv"] = cust["monthly_charges"] * 12 * 1.2

        lp_res = self._quick_solve(cust, budget)
        lp_t = lp_res[lp_res["action"] != "no_action"]
        lp_cost = lp_t["cost"].sum()
        lp_saved = lp_t["expected_saved"].sum()

        actions = {k: v for k, v in self.ACTIONS.items()
                   if k != "no_action"}
        best_action = max(
            actions.items(),
            key=lambda kv: kv[1]["success"] / max(kv[1]["cost"], 1))
        ba_name, ba_info = best_action
        cust_sorted = cust.copy()
        cust_sorted["ev_density"] = (
            cust_sorted["churn_prob"]
            * ba_info["success"]
            * cust_sorted["clv"]
            / max(ba_info["cost"], 1))
        cust_sorted = cust_sorted.sort_values(
            "ev_density", ascending=False)
        greedy_cost, greedy_saved = 0, 0
        greedy_n = 0
        for _, r in cust_sorted.iterrows():
            if greedy_cost + ba_info["cost"] <= budget:
                greedy_cost += ba_info["cost"]
                greedy_saved += (r["churn_prob"]
                                 * ba_info["success"] * r["clv"])
                greedy_n += 1

        rng = np.random.RandomState(self.cfg.SEED)
        perm = rng.permutation(len(cust))
        rand_cost, rand_saved, rand_n = 0, 0, 0
        for i in perm:
            r = cust.iloc[i]
            if rand_cost + ba_info["cost"] <= budget:
                rand_cost += ba_info["cost"]
                rand_saved += (r["churn_prob"]
                               * ba_info["success"] * r["clv"])
                rand_n += 1

        results = pd.DataFrame([
            {"Strategy": "LP Optimal", "Cost": lp_cost,
             "Saved": lp_saved, "Targeted": len(lp_t),
             "ROI": ((lp_saved - lp_cost)
                     / max(lp_cost, 1) * 100)},
            {"Strategy": "Greedy", "Cost": greedy_cost,
             "Saved": greedy_saved, "Targeted": greedy_n,
             "ROI": ((greedy_saved - greedy_cost)
                     / max(greedy_cost, 1) * 100)},
            {"Strategy": "Random", "Cost": rand_cost,
             "Saved": rand_saved, "Targeted": rand_n,
             "ROI": ((rand_saved - rand_cost)
                     / max(rand_cost, 1) * 100)},
        ])
        print(results.to_string(index=False))

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        ax.bar(results["Strategy"], results["ROI"], color=colors)
        for i, v in enumerate(results["ROI"]):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center",
                    fontweight="bold")
        ax.set_ylabel("ROI (%)")
        ax.set_title("Strategy Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.FIG_DIR,
                                 "strategy_compare.png"),
                    dpi=self.cfg.FIG_DPI)
        plt.close()

        results.to_csv(os.path.join(
            self.cfg.RESULT_DIR,
            "strategy_comparison.csv"), index=False)
        return results


# ====================================================================
# SECTION 11 — RESULTS TABLE GENERATOR
# ====================================================================
class TableGen:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def ablation_table(self, cv_store):
        rows = []
        for exp, d in cv_store.items():
            for nm, s in d["summary"].items():
                rows.append({
                    "Experiment": exp,
                    "Model": nm,
                    "F1": (f"{s['F1_mean']:.4f}"
                           f"±{s['F1_std']:.4f}"),
                    "AUC": (f"{s['AUC_mean']:.4f}"
                            f"±{s['AUC_std']:.4f}"),
                    "PR-AUC": (f"{s['PR_AUC_mean']:.4f}"
                               f"±{s['PR_AUC_std']:.4f}"),
                    "MCC": (f"{s['MCC_mean']:.4f}"
                            f"±{s['MCC_std']:.4f}"),
                    "F1_val": s["F1_mean"],
                })
        df = pd.DataFrame(rows)
        df = df.sort_values("F1_val", ascending=False)
        df.drop("F1_val", axis=1, inplace=True)
        print("\n" + "=" * 65)
        print("RESULTS TABLE")
        print("=" * 65)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(self.cfg.RESULT_DIR,
                               "results_table.csv"), index=False)

        latex = df.to_latex(
            index=False, column_format="llcccc",
            caption=("Ablation study results "
                     "(mean ± std over 5-fold CV)."),
            label="tab:ablation")
        path = os.path.join(self.cfg.RESULT_DIR,
                            "results_table.tex")
        with open(path, "w") as f:
            f.write(latex)
        print(f"  LaTeX table → {path}")
        return df


# ====================================================================
# SECTION 12 — MAIN PIPELINE
# ====================================================================
def main():
    t0 = time.time()
    cfg = Config()
    cfg.setup()
    log = Logger()

    print("\n" + "=" * 65)
    print("RUNTIME CONFIGURATION")
    print("=" * 65)
    print(f"  GPU enabled      : {cfg.USE_GPU}")
    if cfg.USE_GPU:
        print(f"  GPU device       : {GPU_INFO['device_name']}")
        print(f"  VRAM             : {GPU_INFO['vram_gb']:.1f} GB")
    print(f"  Embedding device : {cfg.EMB_DEVICE}")
    print(f"  Embedding batch  : {cfg.EMB_BATCH_SIZE}")
    print(f"  Tuning strategy  : {cfg.TUNING_STRATEGY.upper()}")
    if cfg.TUNING_STRATEGY == "optuna":
        print(f"  Optuna trials    : {cfg.OPTUNA_N_TRIALS}")
        print(f"  Optuna timeout   : {cfg.OPTUNA_TIMEOUT}s/model")
        print(f"  Optuna pruning   : {cfg.OPTUNA_PRUNING}")
    print(f"  XGBoost backend  : {cfg.xgb_gpu_params()}")

    # ── 1. DATA ──────────────────────────────────────────────────────
    dp = DataProcessor(cfg)
    df_raw = dp.load()
    log.add("data", "shape_raw", df_raw.shape)

    eda = EDA(cfg)
    eda.run(df_raw.copy())

    df = dp.engineer()
    y = df["churn"].copy()
    log.add("data", "class_dist", y.value_counts().to_dict())
    print(f"\n  Target: {y.value_counts().to_dict()}"
          f"  ratio = 1:{(y==0).sum()/max((y==1).sum(),1):.1f}")

    # ── 2. EMBEDDINGS (GPU-accelerated) ──────────────────────────────
    embedder = Embedder(cfg)
    text = (df["chat_log"] if "chat_log" in df.columns
            else pd.Series([""] * len(df)))

    emb_t0 = time.time()
    emb_dfs = embedder.embed_all(text)
    tfidf_df = embedder.tfidf(text)
    emb_elapsed = time.time() - emb_t0
    print(f"  Total embedding time: {emb_elapsed:.1f}s")
    log.add("embeddings", "time_seconds", emb_elapsed)

    for tag, edf in emb_dfs.items():
        embedder.cluster_plot(edf, y.values, tag)

    # ── 3. BUILD FEATURE SETS ────────────────────────────────────────
    feat_sets = dp.build_feature_sets(emb_dfs, tfidf_df)
    for name, fdf in feat_sets.items():
        print(f"  Feature set {name:30s} → "
              f"{fdf.shape[1]} features")

    # ── 4. FILL NaN & SPLIT ─────────────────────────────────────────
    for name in feat_sets:
        fdf = feat_sets[name]
        for c in fdf.select_dtypes(include=np.number).columns:
            fdf[c].fillna(fdf[c].median(), inplace=True)
        fdf.fillna(0, inplace=True)
        feat_sets[name] = fdf

    first_key = list(feat_sets.keys())[0]
    X_all = feat_sets[first_key]
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=cfg.TEST_SIZE,
        stratify=y, random_state=cfg.SEED)

    # ── 5. TRAINER SETUP & TUNING ────────────────────────────────────
    trainer = Trainer(cfg)

    tune_key = [k for k in feat_sets if "Full" in k or "A4" in k]
    if not tune_key:
        tune_key = [list(feat_sets.keys())[-1]]
    X_tune = feat_sets[tune_key[0]].iloc[idx_train]
    y_tune = y.iloc[idx_train]

    tune_t0 = time.time()
    trainer.tune(X_tune, y_tune)
    tune_elapsed = time.time() - tune_t0
    print(f"  Total tuning time: {tune_elapsed:.1f}s")
    log.add("tuning", "best_params", trainer.best_params)
    log.add("tuning", "time_seconds", tune_elapsed)
    log.add("tuning", "strategy", cfg.TUNING_STRATEGY)
    log.add("tuning", "per_model_times", trainer._tuning_times)

    # ── 6. ABLATION CROSS-VALIDATION ─────────────────────────────────
    print("\n" + "=" * 65)
    print("ABLATION STUDY")
    print("=" * 65)
    for name, fdf in feat_sets.items():
        trainer.evaluate_cv(fdf, y, tag=name)

    # ── 7. EVALUATION VISUALISATIONS ─────────────────────────────────
    ev = Evaluator(cfg)
    ev.ablation_chart(trainer.cv)

    best_f1, best_exp, best_mdl = 0, None, None
    for exp, d in trainer.cv.items():
        for nm, s in d["summary"].items():
            if s["F1_mean"] > best_f1:
                best_f1, best_exp, best_mdl = (
                    s["F1_mean"], exp, nm)
    print(f"\n★ Best config: {best_exp} / {best_mdl}  "
          f"(F1={best_f1:.4f})")
    log.add("best", "experiment", best_exp)
    log.add("best", "model", best_mdl)
    log.add("best", "F1_mean", best_f1)

    best_preds = trainer.cv[best_exp]["preds"]
    ev.roc(best_preds, tag=best_exp)
    ev.pr(best_preds, tag=best_exp)
    ev.cm(best_preds, tag=best_exp)
    ev.calibration(best_preds, tag=best_exp)

    # ── 8. TRAIN FINAL MODEL ─────────────────────────────────────────
    X_best = feat_sets[best_exp]
    Xtr = X_best.iloc[idx_train].copy()
    Xte = X_best.iloc[idx_test].copy()
    ytr = y.iloc[idx_train].copy()
    yte = y.iloc[idx_test].copy()

    model, scaler, Xtr_s, Xte_s, y_pred, y_prob = (
        trainer.train_final(Xtr, ytr, Xte, yte, name=best_mdl))

    joblib.dump(model, os.path.join(cfg.MODEL_DIR,
                                    "best_model.pkl"))
    joblib.dump(scaler, os.path.join(cfg.MODEL_DIR,
                                     "scaler.pkl"))
    print(f"  Model saved → {cfg.MODEL_DIR}")

    ev.learn_curve(
        clone(model).set_params(
            **trainer.best_params.get(best_mdl, {})),
        Xtr_s, ytr, best_mdl)

    opt_thr = ev.opt_threshold(yte, y_prob)
    log.add("best", "optimal_threshold", opt_thr)

    # ── 9. STATISTICAL TESTS ─────────────────────────────────────────
    st = StatTests(cfg)
    best_raw = trainer.cv[best_exp]["raw"]
    best_pred = trainer.cv[best_exp]["preds"]
    stat_res = st.run(best_raw, best_pred)
    log.add("stats", "results", stat_res)

    print(f"\n  Bootstrap CI for {best_mdl} AUC:")
    st.bootstrap_ci(
        best_pred[best_mdl]["yt"],
        best_pred[best_mdl]["yb"],
        roc_auc_score)

    # ── 10. EXPLAINABILITY ────────────────────────────────────────────
    explainer = Explainer(cfg)
    feature_names = X_best.columns.tolist()

    sv = explainer.shap_global(
        model, Xte_s, feature_names, best_mdl)
    explainer.shap_local(
        model, Xte_s, feature_names,
        indices=[0, 1, 2], model_name=best_mdl)
    explainer.lime_compare(
        model, Xte_s, feature_names,
        idx=0, model_name=best_mdl)
    explainer.permutation_importance(
        model, Xte_s, yte, feature_names, best_mdl)

    # ── 11. PRESCRIPTIVE OPTIMISATION ─────────────────────────────────
    opt = Optimizer(cfg)

    cust_opt = pd.DataFrame({
        "customer_idx": np.arange(len(y_prob)),
        "churn_prob": y_prob,
        "monthly_charges": Xte.iloc[:, 0].values,
    })
    if "monthlycharges" in Xte.columns:
        cust_opt["monthly_charges"] = (
            Xte["monthlycharges"].values)

    high_risk = cust_opt[cust_opt["churn_prob"] > 0.5].copy()
    if high_risk.empty:
        high_risk = cust_opt.nlargest(20, "churn_prob").copy()
    if len(high_risk) > 50:
        high_risk = high_risk.nlargest(50, "churn_prob").copy()
    print(f"\n  High-risk customers: {len(high_risk)}")

    opt_result = opt.optimise(high_risk)
    sens_df = opt.sensitivity(high_risk)
    mc = opt.monte_carlo(opt_result)
    strat = opt.compare_strategies(high_risk)

    log.add("optimisation", "high_risk_n", len(high_risk))
    if mc:
        log.add("optimisation", "mc_p_positive",
                mc["p_positive"])

    # ── 12. RESULTS TABLE ─────────────────────────────────────────────
    tg = TableGen(cfg)
    tg.ablation_table(trainer.cv)

    # ── 13. FINAL SUMMARY ────────────────────────────────────────────
    elapsed = time.time() - t0

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Total time     : {elapsed/60:.1f} min "
          f"({elapsed:.0f}s)")
    print(f"  GPU used       : {cfg.USE_GPU}")
    if cfg.USE_GPU:
        print(f"  GPU device     : {GPU_INFO['device_name']}")
    print(f"  Tuning strategy: {cfg.TUNING_STRATEGY}")
    print(f"  Tuning time    : {tune_elapsed:.1f}s")
    print(f"  Best experiment: {best_exp}")
    print(f"  Best model     : {best_mdl}")
    print(f"  CV F1          : {best_f1:.4f}")
    print(f"  Opt threshold  : {opt_thr:.2f}")
    print(f"  Figures        : {cfg.FIG_DIR}/")
    print(f"  Results        : {cfg.RESULT_DIR}/")
    print(f"  Models         : {cfg.MODEL_DIR}/")

    log.add("runtime", "seconds", elapsed)
    log.add("runtime", "gpu_used", cfg.USE_GPU)
    log.add("runtime", "tuning_strategy", cfg.TUNING_STRATEGY)
    log.save(os.path.join(cfg.RESULT_DIR, "experiment_log.json"))

    if cfg.USE_GPU:
        torch.cuda.empty_cache()
        print(f"\n  GPU memory cleared.")

    print("\n✓ All outputs saved. Ready for paper integration.\n")


# ====================================================================
if __name__ == "__main__":
    main()