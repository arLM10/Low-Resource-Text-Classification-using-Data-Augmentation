# Project 14: Low-Resource Text Classification using Data Augmentation
## UG Student Project | Step-by-Step Instructions

---

## 📁 Project Structure

```
project14/
├── data/                    ← Auto-created by Week 1 script
├── results/
│   ├── week1/               ← Plots: class distributions, text lengths
│   └── week3/               ← All experiment results + plots
├── week1/
│   └── dataset_analysis.py  ← Week 1 deliverable
├── week2/
│   └── baseline_model.py    ← Week 2 deliverable
├── week3/
│   ├── experiments.py       ← Week 3 deliverable (main)
│   └── visualize_results.py ← Week 3 plots
├── week4/
│   └── final_paper_template.md  ← Week 4 paper template
└── requirements.txt
```

---

## ⚙️ Setup (Do This First)

### Step 1 – Create a Python environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# OR
venv\Scripts\activate           # Windows
```

### Step 2 – Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 If you are on a machine without a GPU, everything still runs on CPU — it will just be slower.

---

## 📅 WEEK 1 — Dataset Analysis & Literature Survey

### What to run:
```bash
cd project14
python week1/dataset_analysis.py
```

### What it does:
- Downloads AG News and IMDb datasets automatically via HuggingFace
- Simulates low-resource settings (5% of AG News, 4% of IMDb)
- Applies preprocessing (lowercase, tokenize, remove stopwords)
- Creates stratified train/val/test splits (70/15/15)
- Saves processed CSVs to `data/`
- Saves class balance and text length plots to `results/week1/`

### What to submit this week:
1. **Plots** from `results/week1/` — paste these into your report
2. **Dataset statistics** printed to terminal — copy into a table
3. **Literature survey** — write 8 paper summaries based on the references in `week4/final_paper_template.md`. Use Google Scholar to find and read those papers.
4. **Problem statement** — write 1 paragraph: "In low-resource NLP, limited labeled data causes X. We address this with Y."

---

## 📅 WEEK 2 — Solution & Methodology

### What to run:
```bash
python week2/baseline_model.py
```

### What it does:
- Builds the Vocabulary from training data
- Trains a TextCNN (Kim, 2014) baseline model
- Runs one seed baseline vs. EDA-augmented comparison
- Prints test accuracy and F1

### What to submit this week:
1. **Architecture diagram** — draw the TextCNN by hand or in draw.io:
   - Input → Embedding (100d) → Conv filters [2,3,4] × 128 → MaxPool → Concat → Dropout → FC
2. **Augmentation examples table** — for 2 sample sentences from your data, show what each EDA operation produces (SR, RI, RS, RD). Just run a few lines interactively:
   ```python
   from week2.baseline_model import *
   text = "This movie was absolutely wonderful"
   print(synonym_replacement(text))
   print(random_deletion(text))
   ```
3. **Methodology writeup** — describe your architecture, each augmentation method, and your experimental plan (what metrics, what hyperparameters, what comparisons).

---

## 📅 WEEK 3 — Implementation & Experiments

### Step 1: Run all experiments (this takes time!)
```bash
python week3/experiments.py
```

> ⏱️ **Expected time:** 20–60 minutes on CPU. On GPU: 5–15 min.  
> This runs every configuration × 3 seeds and saves to `results/week3/all_results.csv`

### Step 2: Generate all plots
```bash
python week3/visualize_results.py
```

### What to submit this week:
1. **`results/week3/all_results.csv`** — your full results table
2. **4 plots** from `results/week3/`:
   - `plot_pruning_acc.png`
   - `plot_quantization.png`
   - `plot_coreset_methods.png`
   - `plot_heatmap_summary.png`
3. **Code** — submit all `.py` files with a `README` explaining how to reproduce
4. **Seed reproducibility note** — confirm you used seeds [42, 123, 7] everywhere

### 🔑 Key things the rubric checks:
- ✅ Baseline trained and evaluated
- ✅ Pruning at 5%, 10%, 25%, 50%, 90%
- ✅ Quantization at 8-bit and 4-bit
- ✅ Coreset at ≥3 sizes (10%, 25%, 50%)
- ✅ All 4 coreset methods (random, k-center, gradient, proposed)
- ✅ Combined configuration (aug + compression + coreset)
- ✅ Multiple seeds → mean ± std

---

## 📅 WEEK 4 — Final Paper + Demo

### Step 1: Fill in your paper
Open `week4/final_paper_template.md` and:
- Replace all `X.XXX` placeholders with your actual numbers from `results/week3/all_results.csv`
- Fill in the **Discussion** section with your interpretation
- Update the **Abstract** with your actual findings
- Add your name and university

### Step 2: Export to PDF
Option A (recommended): Copy the markdown into Overleaf → use ACL or IEEE conference template  
Option B: Use VS Code + Markdown PDF extension  
Option C: Convert with pandoc:
```bash
pandoc week4/final_paper_template.md -o week4/final_paper.pdf
```

### Step 3: Prepare Demo
Your demo should show:
1. Run `python week3/experiments.py` live (or show a saved run)
2. Show the results CSV and plots
3. Live example: feed a sentence → get prediction, before and after augmentation

### What to submit this week:
1. **Final paper PDF** (conference format, 6–8 pages)
2. **All code** (zipped or GitHub link)
3. **`data/` splits** (CSV files for reproducibility)
4. **Live demo** during presentation

---

## 📊 Summary: What Each File Produces

| File | Output | Used In |
|---|---|---|
| `week1/dataset_analysis.py` | data/ CSVs, results/week1/ plots | Week 1 report |
| `week2/baseline_model.py` | Model functions, smoke-test results | Week 2 report |
| `week3/experiments.py` | results/week3/all_results.csv | Week 3 + Paper |
| `week3/visualize_results.py` | 4 result plots | Week 3 + Paper |
| `week4/final_paper_template.md` | Paper draft | Week 4 submission |

---

## 🧪 Reproducibility Checklist

- [ ] All random seeds set to [42, 123, 7]
- [ ] `data/` CSVs committed (or reproducible from script)
- [ ] `requirements.txt` pinned
- [ ] Results reported as mean ± std
- [ ] Code runs end-to-end with: `python week1/dataset_analysis.py && python week3/experiments.py`
