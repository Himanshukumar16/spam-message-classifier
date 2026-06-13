import os
import re
import sys
import warnings
import argparse
# import time
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
# from rich.rule import Rule
from rich.prompt import Prompt
from rich.theme import Theme
from rich import box
warnings.filterwarnings("ignore")
matplotlib.use("Agg")


#  2. DATA LOADING
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        console.print(f"[error]Dataset not found at '{path}'.[/error]\n"
                      "Place spam_dataset.csv in the same folder as main.py.")
        sys.exit(1)

    df = pd.read_csv(path)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    # Accept various column naming conventions
    if "label" in df.columns and "message" in df.columns:
        pass
    elif "v1" in df.columns and "v2" in df.columns:
        df.rename(columns={"v1": "label", "v2": "message"}, inplace=True)
    elif "category" in df.columns and "message" in df.columns:
        df.rename(columns={"category": "label"}, inplace=True)
    else:
        console.print("[error]Could not detect label/message columns.[/error]")
        sys.exit(1)

    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()

    # Map to 0/1
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    # Clean text
    df["clean"] = df["message"].apply(clean_text)

    return df
