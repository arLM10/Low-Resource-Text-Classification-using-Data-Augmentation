"""
Week 3: Visualize Experiment Results
Generates all plots needed for Week 4 paper.
Run AFTER experiments.py has completed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

os.makedirs("results/week3", exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

df = pd.read_csv("results/week3/all_results.csv")

def bar_with_errorbars(subset, title, out_path, metric="acc"):
    fig, ax = plt.subplots(figsize=(max(8, len(subset) * 0.9), 5))
    x       = range(len(subset))
    means   = subset[f"{metric}_mean"].values
    stds    = subset[f"{metric}_std"].values
    colors  = ["#2C7BB6" if i == 0 else "#D7191C" if "Combined" in subset["label"].iloc[i]
               else "#ABD9E9" for i in range(len(subset))]

    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(subset["label"].tolist(), rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Weighted F1" if metric == "f1" else "Accuracy")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(max(0, means.min() - 0.1), min(1.0, means.max() + 0.12))

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# ── Plot 1: Pruning ─────────────────────────────────────────────────────────────
prune_rows = df[df["label"].str.startswith("Baseline") |
                df["label"].str.startswith("Prune")]
bar_with_errorbars(prune_rows, "Effect of Magnitude Pruning on Test Accuracy",
                   "results/week3/plot_pruning_acc.png", metric="acc")
bar_with_errorbars(prune_rows, "Effect of Magnitude Pruning on Weighted F1",
                   "results/week3/plot_pruning_f1.png", metric="f1")

# ── Plot 2: Quantization ────────────────────────────────────────────────────────
quant_rows = df[df["label"].str.startswith("Baseline") |
                df["label"].str.startswith("Quant")]
bar_with_errorbars(quant_rows, "Effect of Quantization on Test Accuracy",
                   "results/week3/plot_quantization.png", metric="acc")

# ── Plot 3: Coreset (Accuracy by fraction, per method) ──────────────────────────
core_rows = df[df["label"].str.startswith("Coreset")]
methods   = ["random", "kcenter", "gradient", "proposed"]
fracs     = [10, 25, 50]
colors    = ["#4DAF4A", "#377EB8", "#FF7F00", "#E41A1C"]

fig, ax = plt.subplots(figsize=(8, 5))
for method, color in zip(methods, colors):
    sub = core_rows[core_rows["label"].str.contains(method)]
    sub = sub.copy()
    sub["frac"] = sub["label"].str.extract(r"(\d+)%").astype(int)
    sub = sub.sort_values("frac")
    ax.errorbar(sub["frac"], sub["acc_mean"], yerr=sub["acc_std"],
                label=method, color=color, marker="o", capsize=4, linewidth=2)

base_acc = df[df["label"] == "Baseline"]["acc_mean"].values[0]
ax.axhline(base_acc, linestyle="--", color="gray", label="Baseline")
ax.set_xlabel("Coreset Fraction (%)")
ax.set_ylabel("Test Accuracy")
ax.set_title("Coreset Selection Methods vs. Accuracy", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("results/week3/plot_coreset_methods.png", dpi=150)
plt.close()
print("Saved: results/week3/plot_coreset_methods.png")

# ── Plot 4: Full Comparison Heatmap ────────────────────────────────────────────
pivot_labels = ["Baseline", "Aug_EDA",
                "Prune_25%", "Prune_50%",
                "Quant_8bit", "Quant_4bit",
                "Coreset_random_50%", "Coreset_kcenter_50%",
                "Coreset_gradient_50%", "Coreset_proposed_50%",
                "Combined_Aug+Prune25+Coreset50",
                "Combined_Aug+Quant8+Coreset50"]

sub = df[df["label"].isin(pivot_labels)].set_index("label")
sub = sub.reindex(pivot_labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, metric, title in zip(axes,
                              ["acc_mean", "f1_mean"],
                              ["Test Accuracy", "Weighted F1"]):
    vals   = sub[[metric]].values
    im     = ax.imshow(vals, cmap="RdYlGn", vmin=vals.min()-0.02, vmax=vals.max()+0.02,
                       aspect="auto")
    ax.set_yticks(range(len(pivot_labels)))
    ax.set_yticklabels(pivot_labels, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels([title])
    ax.set_title(title, fontweight="bold")
    for i, v in enumerate(vals.flatten()):
        ax.text(0, i, f"{v:.3f}", ha="center", va="center",
                color="black", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04)

plt.suptitle("Experiment Summary Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/week3/plot_heatmap_summary.png", dpi=150)
plt.close()
print("Saved: results/week3/plot_heatmap_summary.png")

print("\n✅ All plots saved to results/week3/")
