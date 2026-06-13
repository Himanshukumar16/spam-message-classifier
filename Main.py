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

#  1. TEXT PREPROCESSING

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+",              " urltoken ",  text)
    text = re.sub(r"\b\d{5,}\b",                    " phonetoken ", text)
    text = re.sub(r"[£$€]\d+[\d,]*",                " moneytoken ", text)
    text = re.sub(r"[^a-z0-9\s]",                   " ",           text)
    text = re.sub(r"\s+",                            " ",           text).strip()
    return text

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

#  3. MODEL BUILDING

def build_pipeline() -> Pipeline:
    tfidf = TfidfVectorizer(
        analyzer       = "char_wb",     # character n-grams handle typos/obfuscation
        ngram_range    = (2, 5),
        max_features   = 80_000,
        sublinear_tf   = True,
        min_df         = 1,
        strip_accents  = "unicode",
    )

    # Word-level vectorised blended via manual feature concat
    tfidf_word = TfidfVectorizer(
        analyzer      = "word",
        ngram_range   = (1, 2),
        max_features  = 40_000,
        sublinear_tf  = True,
        min_df        = 1,
        strip_accents = "unicode",
    )

    from sklearn.pipeline import FeatureUnion

    features = FeatureUnion([
        ("char_tfidf", tfidf),
        ("word_tfidf", tfidf_word),
    ])

    clf = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, class_weight="balanced"),
        cv=3,
        method="sigmoid",
    )

    pipe = Pipeline([
        ("features", features),
        ("clf",      clf),
    ])
    return pipe

#  4. EVALUATION HELPERS

def print_banner():
    console.print(
        Panel.fit(
            "SPAM EMAIL CLASSIFIER",
            border_style="white",
            padding=(1, 4)
        )
    )


def print_dataset_stats(df: pd.DataFrame):
    spam_n = (df["target"] == 1).sum()
    ham_n  = (df["target"] == 0).sum()
    total  = len(df)

    t = Table(title="Dataset Overview", box=box.ROUNDED, border_style="cyan")
    t.add_column("Split",         style="bold")
    t.add_column("Count",         justify="right")
    t.add_column("Percentage",    justify="right")

    t.add_row("Total messages", str(total),  "100.0%")
    t.add_row("[green]Ham (legit)[/green]",  str(ham_n),
              f"{ham_n/total*100:.1f}%")
    t.add_row("[red]Spam[/red]",            str(spam_n),
              f"{spam_n/total*100:.1f}%")

    # console.print(t)
    # console.print()


def print_metrics_table(metrics: dict):
    t = Table(title="📈 Classification Metrics", box=box.DOUBLE_EDGE,
              border_style="green")
    t.add_column("Metric",    style="bold white")
    t.add_column("Score",     justify="right", style="bold")
    t.add_column("Status",    justify="center")

    thresholds = {
        "Accuracy":  0.95,
        "F1-Score":  0.95,
        "Recall":    0.90,
        "Precision": 0.95,
        "ROC-AUC":   0.97,
    }

    for name, val in metrics.items():
        threshold = thresholds.get(name, 0.90)
        passed    = val >= threshold
        score_str = f"{val*100:.2f}%"
        status    = "[green]✔ PASS[/green]" if passed else "[red]✘ FAIL[/red]"
        color     = "green" if passed else "red"
        t.add_row(name, f"[{color}]{score_str}[/{color}]", status)

    # console.print(t)
    # console.print()


def print_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred,
                                   target_names=["Ham", "Spam"])
    # console.print(Panel(
    #     f"[dim]{report}[/dim]",
    #     title="[bold]Full Classification Report[/bold]",
    #     border_style="blue",
    # ))


def plot_confusion_matrix(y_test, y_pred, save_path: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")

    labels = ["Ham", "Spam"]
    cmap   = sns.color_palette("Blues", as_cmap=True)

    # --- Raw counts ---
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=axes[0],
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="#30363d",
                annot_kws={"size": 18, "color": "white"})
    axes[0].set_title("Confusion Matrix (Counts)",
                      color="white", fontsize=14, pad=12)
    axes[0].set_xlabel("Predicted Label", color="#8b949e")
    axes[0].set_ylabel("True Label",      color="#8b949e")
    axes[0].tick_params(colors="white")

    # --- Normalised ---
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap=cmap, ax=axes[1],
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="#30363d",
                annot_kws={"size": 16, "color": "white"})
    axes[1].set_title("Confusion Matrix (Normalised)",
                      color="white", fontsize=14, pad=12)
    axes[1].set_xlabel("Predicted Label", color="#8b949e")
    axes[1].set_ylabel("True Label",      color="#8b949e")
    axes[1].tick_params(colors="white")

    # Stat annotations
    tn, fp, fn, tp = cm.ravel()
    stats = (f"TN={tn} | FP={fp} | FN={fn} | TP={tp}\n"
             f"False-Positive Rate: {fp/(fp+tn)*100:.2f}%   "
             f"False-Negative Rate: {fn/(fn+tp)*100:.2f}%")
    fig.text(0.5, 0.01, stats, ha="center", color="#8b949e", fontsize=10)

    plt.suptitle(" Spam Classifier — Confusion Matrix",
                 color="white", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    # console.print(f"[info]Confusion matrix saved → [underline]{save_path}[/underline][/info]")


def run_cross_validation(pipe: Pipeline, X: pd.Series, y: pd.Series):
    # console.print("[info]Running 5-fold stratified cross-validation…[/info]")
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1", n_jobs=-1)

    t = Table(title=" 5-Fold Cross-Validation (F1)", box=box.SIMPLE)
    t.add_column("Fold",  justify="center")
    t.add_column("F1",    justify="right", style="cyan")

    for i, s in enumerate(scores, 1):
        t.add_row(str(i), f"{s*100:.2f}%")

    t.add_row("[bold]Mean[/bold]",
              f"[bold green]{scores.mean()*100:.2f}%[/bold green]")
    t.add_row("[bold]Std[/bold]",
              f"[bold]{scores.std()*100:.2f}%[/bold]")

    # console.print(t)
    # console.print()

    #  5. INTERACTIVE PREDICTOR

def interactive_predictor(pipe: Pipeline):
    # console.print(Rule("[bold cyan] Interactive Spam Detector[/bold cyan]"))
    console.print("Type a message and press Enter to classify it.")
    console.print("[dim]Type [bold]quit[/bold] or [bold]exit[/bold] to stop.[/dim]\n")

    example_messages = [
        "WINNER!! You have been selected to receive a £1000 prize! Call 09061701461 now!",
        "Hey, are you coming to the office party tonight?",
        "Congratulations! Your phone number has WON £500,000. Claim at once!",
        "Can you pick up some milk on your way home?",
        "FREE entry to win a holiday! Text WIN to 87575 now! T&Cs apply.",
    ]

    console.print("[dim]── Example messages you can try ──[/dim]")
    for i, msg in enumerate(example_messages, 1):
        console.print(f"  [cyan]{i}.[/cyan] {msg[:80]}{'…' if len(msg)>80 else ''}")
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold white]Enter message[/bold white]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[warn]Exiting…[/warn]")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("[info]Thank You![/info]")
            break

        if not user_input.strip():
            continue

        cleaned    = clean_text(user_input)
        proba      = pipe.predict_proba([cleaned])[0]
        prediction = pipe.predict([cleaned])[0]

        spam_prob = proba[1] * 100
        ham_prob  = proba[0] * 100

        if prediction == 1:
            verdict = Panel(
                f"[bold red] SPAM DETECTED[/bold red]\n\n"
                f"Spam probability : [red]{spam_prob:.1f}%[/red]\n"
                f"Ham  probability : [green]{ham_prob:.1f}%[/green]",
                border_style="red",
                padding=(0, 1),
            )
        else:
            verdict = Panel(
                f"[bold green] LEGITIMATE (Ham)[/bold green]\n\n"
                f"Ham  probability : [green]{ham_prob:.1f}%[/green]\n"
                f"Spam probability : [red]{spam_prob:.1f}%[/red]",
                border_style="green",
                padding=(0, 1),
            )

        console.print(verdict)
        console.print()

