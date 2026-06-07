"""
Week 1: Dataset Analysis & Literature Survey
Project 14: Low-Resource Text Classification using Data Augmentation
UG Student Project

This script:
- Downloads/loads two publicly available datasets
- Performs preprocessing, train/val/test splits
- Analyzes class balance
- Visualizes statistics
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEEDS = [42, 123, 7]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# ─── Dataset 1: AG News (Topic Classification) ─────────────────────────────────
def load_agnews():
    print("\n[Dataset 1] Loading AG News...")
    dataset = load_dataset("ag_news")
    train_df = pd.DataFrame(dataset["train"])
    test_df  = pd.DataFrame(dataset["test"])
    train_df.rename(columns={"text": "text", "label": "label"}, inplace=True)
    test_df.rename(columns={"text": "text", "label": "label"}, inplace=True)
    label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    train_df["label_name"] = train_df["label"].map(label_names)
    test_df["label_name"]  = test_df["label"].map(label_names)
    return train_df, test_df, label_names

# ─── Dataset 2: IMDb Sentiment (Binary Classification) ─────────────────────────
def load_imdb():
    print("\n[Dataset 2] Loading IMDb Sentiment...")
    dataset = load_dataset("imdb")
    train_df = pd.DataFrame(dataset["train"])
    test_df  = pd.DataFrame(dataset["test"])
    label_names = {0: "Negative", 1: "Positive"}
    train_df["label_name"] = train_df["label"].map(label_names)
    test_df["label_name"]  = test_df["label"].map(label_names)
    return train_df, test_df, label_names

# ─── Low-Resource Simulation ────────────────────────────────────────────────────
def simulate_low_resource(df, fraction=0.1, seed=42):
    """Sample a small fraction per class to simulate low-resource setting."""
    set_seed(seed)
    sampled = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=seed)
    )
    return sampled.reset_index(drop=True)

# ─── Preprocessing ──────────────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

def preprocess_df(df):
    df = df.copy()
    df["clean_text"] = df["text"].apply(preprocess_text)
    df["text_len"]   = df["text"].apply(lambda x: len(x.split()))
    df["clean_len"]  = df["clean_text"].apply(lambda x: len(x.split()))
    return df

# ─── Train / Val / Test Splits ──────────────────────────────────────────────────
def make_splits(df, seed=42):
    train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=seed)
    val,  test  = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=seed)
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

# ─── Analysis & Plots ───────────────────────────────────────────────────────────
def class_balance_plot(df, title, out_path):
    counts = df["label_name"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="Blues_d", ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

def text_length_plot(df, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["text_len"].hist(bins=50, ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("Original Text Length")
    axes[0].set_xlabel("Words")
    df["clean_len"].hist(bins=50, ax=axes[1], color="coral", edgecolor="white")
    axes[1].set_title("Cleaned Text Length")
    axes[1].set_xlabel("Words")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

def print_stats(df, name):
    print(f"\n{'─'*50}")
    print(f"  Dataset: {name}")
    print(f"  Total samples : {len(df)}")
    print(f"  Unique classes: {df['label'].nunique()}")
    print(f"  Class dist.   :\n{df['label_name'].value_counts().to_string()}")
    print(f"  Avg text len  : {df['text_len'].mean():.1f} words")
    print(f"  Median len    : {df['text_len'].median():.1f} words")
    print(f"{'─'*50}")

# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results/week1", exist_ok=True)

    # ── Dataset 1: AG News ──────────────────────────────────────────────────────
    ag_train, ag_test, ag_labels = load_agnews()
    ag_low   = simulate_low_resource(ag_train, fraction=0.05, seed=SEEDS[0])  # ~5% = low resource
    ag_train = preprocess_df(ag_train)
    ag_low   = preprocess_df(ag_low)

    print_stats(ag_train, "AG News (Full)")
    print_stats(ag_low,   "AG News (Low-Resource 5%)")

    class_balance_plot(ag_train, "AG News – Full Class Distribution",
                       "results/week1/agnews_full_class_dist.png")
    class_balance_plot(ag_low,   "AG News – Low-Resource Class Distribution",
                       "results/week1/agnews_low_class_dist.png")
    text_length_plot(ag_low,     "AG News – Text Length Distribution",
                     "results/week1/agnews_text_len.png")

    train_ag, val_ag, test_ag = make_splits(ag_low, seed=SEEDS[0])
    train_ag.to_csv("data/agnews_train.csv", index=False)
    val_ag.to_csv("data/agnews_val.csv",     index=False)
    test_ag.to_csv("data/agnews_test.csv",   index=False)
    print("  AG News splits saved to data/")

    # ── Dataset 2: IMDb ──────────────────────────────────────────────────────────
    imdb_train, imdb_test, imdb_labels = load_imdb()
    imdb_low = simulate_low_resource(imdb_train, fraction=0.04, seed=SEEDS[0])  # ~4% = low resource
    imdb_low = preprocess_df(imdb_low)

    print_stats(imdb_low, "IMDb (Low-Resource 4%)")

    class_balance_plot(imdb_low, "IMDb – Low-Resource Class Distribution",
                       "results/week1/imdb_class_dist.png")
    text_length_plot(imdb_low,   "IMDb – Text Length Distribution",
                     "results/week1/imdb_text_len.png")

    train_im, val_im, test_im = make_splits(imdb_low, seed=SEEDS[0])
    train_im.to_csv("data/imdb_train.csv", index=False)
    val_im.to_csv("data/imdb_val.csv",     index=False)
    test_im.to_csv("data/imdb_test.csv",   index=False)
    print("  IMDb splits saved to data/")

    print("\n✅ Week 1 Complete. Check results/week1/ for plots.")
