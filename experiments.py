"""
Week 3: Full Experiments
Project 14: Low-Resource Text Classification using Data Augmentation

Experiments:
  A. Baseline (no aug, no compression)
  B. Augmentation only (EDA, back-translation proxy, BERT insert)
  C. Model compression: magnitude pruning (5/10/25/50/90%) + quantization (8-bit, 4-bit)
  D. Coreset selection (10/25/50%) – random, k-center, gradient-based, proposed
  E. Combined: augmentation + compression + coreset

All experiments run with SEEDS = [42, 123, 7] → report mean ± std
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week2'))

import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings("ignore")

from baseline_model import (
    set_seed, Vocabulary, TextDataset, TextCNN,
    train_epoch, evaluate, augment_dataset, run_pipeline,
    SEEDS
)

os.makedirs("results/week3", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 1: MODEL COMPRESSION ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def magnitude_prune(model, prune_ratio):
    """
    Magnitude-based unstructured pruning.
    Zeros out the bottom `prune_ratio` fraction of weights in Conv and Linear layers.
    """
    model = copy.deepcopy(model)
    all_weights = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            all_weights.append(module.weight.data.abs().flatten())

    all_weights = torch.cat(all_weights)
    threshold   = torch.quantile(all_weights, prune_ratio)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            mask = module.weight.data.abs() > threshold
            module.weight.data *= mask

    sparsity = (all_weights < threshold).float().mean().item()
    return model, sparsity

def count_nonzero_params(model):
    total = 0
    nonzero = 0
    for p in model.parameters():
        total   += p.numel()
        nonzero += (p != 0).sum().item()
    return nonzero, total

def quantize_model_8bit(model):
    """
    Simulate 8-bit quantization via dynamic quantization.
    Torch supports this for Linear layers natively.
    """
    q_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(model.cpu()),
        {nn.Linear},
        dtype=torch.qint8
    )
    return q_model

def simulate_4bit_quantization(model):
    """
    Simulate 4-bit by clamping and rounding weights to 16 discrete levels.
    (True 4-bit inference requires custom kernels; this simulates the precision loss.)
    """
    model = copy.deepcopy(model)
    with torch.no_grad():
        for p in model.parameters():
            min_v, max_v = p.data.min(), p.data.max()
            scale  = (max_v - min_v) / 15.0 + 1e-8
            p.data = torch.round((p.data - min_v) / scale) * scale + min_v
    return model

# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 2: CORESET SELECTION ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def random_coreset(df, fraction, seed=42):
    """Random uniform sampling per class."""
    set_seed(seed)
    return df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=seed)
    ).reset_index(drop=True)

def kcenter_coreset(df, fraction, seed=42):
    """
    K-Center Greedy coreset (Sener & Savarese, 2018).
    Represents training data by choosing samples that minimize
    the maximum distance to their nearest center.
    Uses TF-IDF features for efficiency.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import pairwise_distances

    set_seed(seed)
    n_select = max(1, int(len(df) * fraction))
    vectorizer = TfidfVectorizer(max_features=512, sublinear_tf=True)
    X = vectorizer.fit_transform(df["clean_text"].tolist()).toarray()

    selected = [random.randint(0, len(X) - 1)]
    min_dists = np.full(len(X), np.inf)

    while len(selected) < n_select:
        last = selected[-1]
        dists = pairwise_distances(X[last:last+1], X, metric="euclidean")[0]
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -np.inf  # exclude already selected
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

    return df.iloc[selected].reset_index(drop=True)

def gradient_coreset(df, fraction, vocab, num_classes, device, seed=42):
    """
    Gradient-based coreset (Craig, Mirzasoleiman et al. 2020 proxy).
    Selects samples whose gradient norms are largest (most informative).
    """
    set_seed(seed)
    n_select = max(1, int(len(df) * fraction))

    model = TextCNN(len(vocab), 100, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    ds     = TextDataset(df["clean_text"].tolist(), df["label"].tolist(), vocab)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    grad_norms = []
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        grad_norms.append(total_norm)

    top_idx = np.argsort(grad_norms)[-n_select:]
    return df.iloc[top_idx].reset_index(drop=True)

def proposed_coreset(df, fraction, seed=42):
    """
    ★ Proposed Method: Class-Balanced Uncertainty Sampling
    ─────────────────────────────────────────────────────────
    Idea: Train a lightweight TF-IDF + Logistic Regression classifier.
          Select samples with HIGHEST entropy (most uncertain predictions),
          but enforce strict per-class balance to handle label skew.

    Motivation: In low-resource settings, the model is most confused on
    boundary samples. Selecting those for training forces the model to
    learn more discriminative features, while class balancing prevents
    majority class dominance.

    This is a novel combination of:
      - Uncertainty sampling (active learning literature)
      - Stratified selection (fairness-aware ML)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import entropy as scipy_entropy

    set_seed(seed)
    n_select = max(1, int(len(df) * fraction))
    n_per_class = max(1, n_select // df["label"].nunique())

    vectorizer = TfidfVectorizer(max_features=1024, sublinear_tf=True)
    X = vectorizer.fit_transform(df["clean_text"].tolist())
    y = df["label"].values

    clf = LogisticRegression(max_iter=200, random_state=seed)
    clf.fit(X, y)
    proba    = clf.predict_proba(X)
    entropies = np.array([scipy_entropy(p) for p in proba])

    selected = []
    for label in df["label"].unique():
        idx_class = np.where(y == label)[0]
        ent_class = entropies[idx_class]
        top_n     = idx_class[np.argsort(ent_class)[-n_per_class:]]
        selected.extend(top_n.tolist())

    return df.iloc[selected].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 3: MULTI-SEED EXPERIMENT RUNNER ───────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_seeds(train_df, val_df, test_df, num_classes,
              augment=False, prune_ratio=None, quantize_bits=None,
              coreset_fn=None, coreset_frac=None,
              num_epochs=10, label="experiment"):
    """
    Run a configuration across all 3 seeds.
    Returns mean ± std of accuracy and F1.
    """
    accs, f1s = [], []

    for seed in SEEDS:
        set_seed(seed)
        tr = train_df.copy()

        # Step 1: Coreset selection
        if coreset_fn is not None and coreset_frac is not None:
            if coreset_fn == "gradient":
                vocab_tmp = Vocabulary()
                vocab_tmp.build(tr["clean_text"].tolist())
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tr = gradient_coreset(tr, coreset_frac, vocab_tmp, num_classes, device, seed)
            elif coreset_fn == "kcenter":
                tr = kcenter_coreset(tr, coreset_frac, seed)
            elif coreset_fn == "proposed":
                tr = proposed_coreset(tr, coreset_frac, seed)
            else:
                tr = random_coreset(tr, coreset_frac, seed)

        # Step 2: Augmentation
        if augment:
            tr = augment_dataset(tr, num_aug=4, alpha=0.1, seed=seed)

        # Step 3: Build model + train
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab  = Vocabulary()
        vocab.build(tr["clean_text"].tolist())

        def make_loader(df, shuffle=False):
            ds = TextDataset(df["clean_text"].tolist(), df["label"].tolist(), vocab)
            return DataLoader(ds, batch_size=64, shuffle=shuffle)

        train_loader = make_loader(tr, shuffle=True)
        val_loader   = make_loader(val_df)
        test_loader  = make_loader(test_df)

        model     = TextCNN(len(vocab), 100, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_val, best_state = 0, None
        for epoch in range(1, num_epochs + 1):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_acc, _ = evaluate(model, val_loader, device)
            if val_acc > best_val:
                best_val   = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)

        # Step 4: Compression (post-training)
        if prune_ratio is not None:
            model, sparsity = magnitude_prune(model, prune_ratio)
            model = model.to(device)

        if quantize_bits == 8:
            model = quantize_model_8bit(model)
            test_loader = make_loader(test_df)   # CPU loader for quantized

        elif quantize_bits == 4:
            model = simulate_4bit_quantization(model).to(device)

        acc, f1 = evaluate(model, test_loader,
                           torch.device("cpu") if quantize_bits == 8 else device)
        accs.append(acc)
        f1s.append(f1)
        print(f"    [{label}] seed={seed} | Acc={acc:.4f} F1={f1:.4f}")

    result = {
        "label":    label,
        "acc_mean": np.mean(accs),
        "acc_std":  np.std(accs),
        "f1_mean":  np.mean(f1s),
        "f1_std":   np.std(f1s),
    }
    print(f"  ► {label}: Acc={result['acc_mean']:.4f}±{result['acc_std']:.4f}  "
          f"F1={result['f1_mean']:.4f}±{result['f1_std']:.4f}")
    return result

# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 4: MAIN EXPERIMENT LOOP ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load AG News splits (primary dataset)
    train_df = pd.read_csv("data/agnews_train.csv")
    val_df   = pd.read_csv("data/agnews_val.csv")
    test_df  = pd.read_csv("data/agnews_test.csv")
    num_classes = train_df["label"].nunique()

    results = []

    # ── A. Baseline ────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("A. BASELINE (no aug, no compression, full data)")
    print("═"*60)
    results.append(run_seeds(train_df, val_df, test_df, num_classes,
                             label="Baseline"))

    # ── B. Data Augmentation ───────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("B. DATA AUGMENTATION (EDA)")
    print("═"*60)
    results.append(run_seeds(train_df, val_df, test_df, num_classes,
                             augment=True, label="Aug_EDA"))

    # ── C. Compression: Pruning ────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("C. PRUNING EXPERIMENTS")
    print("═"*60)
    for ratio in [0.05, 0.10, 0.25, 0.50, 0.90]:
        results.append(run_seeds(train_df, val_df, test_df, num_classes,
                                 prune_ratio=ratio,
                                 label=f"Prune_{int(ratio*100)}%"))

    # ── C. Compression: Quantization ──────────────────────────────────────────
    print("\n" + "═"*60)
    print("C. QUANTIZATION EXPERIMENTS")
    print("═"*60)
    results.append(run_seeds(train_df, val_df, test_df, num_classes,
                             quantize_bits=8, label="Quant_8bit"))
    results.append(run_seeds(train_df, val_df, test_df, num_classes,
                             quantize_bits=4, label="Quant_4bit"))

    # ── D. Coreset Selection ───────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("D. CORESET SELECTION")
    print("═"*60)
    for frac in [0.10, 0.25, 0.50]:
        for method in ["random", "kcenter", "gradient", "proposed"]:
            results.append(run_seeds(
                train_df, val_df, test_df, num_classes,
                coreset_fn=method, coreset_frac=frac,
                label=f"Coreset_{method}_{int(frac*100)}%"
            ))

    # ── E. Combined: Aug + Prune + Coreset ────────────────────────────────────
    print("\n" + "═"*60)
    print("E. COMBINED: Augmentation + Pruning + Coreset")
    print("═"*60)
    results.append(run_seeds(
        train_df, val_df, test_df, num_classes,
        augment=True, prune_ratio=0.25, coreset_fn="proposed", coreset_frac=0.50,
        label="Combined_Aug+Prune25+Coreset50"
    ))
    results.append(run_seeds(
        train_df, val_df, test_df, num_classes,
        augment=True, quantize_bits=8, coreset_fn="proposed", coreset_frac=0.50,
        label="Combined_Aug+Quant8+Coreset50"
    ))

    # ── Save Results ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/week3/all_results.csv", index=False)
    print("\n\n✅ All results saved to results/week3/all_results.csv")
    print(results_df[["label", "acc_mean", "acc_std", "f1_mean", "f1_std"]].to_string(index=False))
