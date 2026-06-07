"""
Week 2: Baseline Neural Model + Data Augmentation Methodology
Project 14: Low-Resource Text Classification using Data Augmentation

Baseline: TextCNN (Kim, 2014)
Augmentation methods:
  1. Random Deletion
  2. Random Swap
  3. Synonym Replacement (EDA)
  4. Back-Translation (MarianMT)
  5. Contextual Word Insertion (BERT masked LM)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import nltk
from nltk.corpus import wordnet, stopwords
nltk.download("wordnet",   quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("omw-1.4",   quiet=True)

# ─── Seeds ──────────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 7]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ─── Tokenizer / Vocab ──────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, max_vocab=30000):
        self.max_vocab = max_vocab
        self.word2idx  = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word  = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        from collections import Counter
        counter = Counter(w for t in texts for w in t.split())
        for word, _ in counter.most_common(self.max_vocab - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

    def encode(self, text, max_len=128):
        tokens = text.split()[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]
        ids   += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)

# ─── Dataset ────────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.data    = [vocab.encode(t, max_len) for t in texts]
        self.labels  = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))

# ─── TextCNN Model ───────────────────────────────────────────────────────────────
class TextCNN(nn.Module):
    """
    Kim (2014) Convolutional Neural Networks for Sentence Classification.
    Filter sizes: [2, 3, 4] with 128 filters each.
    """
    def __init__(self, vocab_size, embed_dim, num_classes,
                 filter_sizes=(2, 3, 4), num_filters=128, dropout=0.5):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs       = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes
        ])
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)       # (B, E, L)
        x = [F.relu(conv(x)) for conv in self.convs]  # list of (B, F, L')
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        x = torch.cat(x, dim=1)                        # (B, 3F)
        x = self.dropout(x)
        return self.fc(x)

# ─── Training / Evaluation ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).argmax(1).cpu().tolist()
            preds  += out
            labels += yb.tolist()
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    return acc, f1

# ─── Data Augmentation Methods ──────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            s = lemma.name().replace("_", " ")
            if s.lower() != word.lower():
                synonyms.add(s)
    return list(synonyms)

def synonym_replacement(text, n=1):
    """EDA: Replace n random non-stopword tokens with a synonym."""
    words = text.split()
    candidates = [w for w in words if w not in stop_words and get_synonyms(w)]
    if not candidates:
        return text
    for word in random.sample(candidates, min(n, len(candidates))):
        syns = get_synonyms(word)
        if syns:
            words = [random.choice(syns) if w == word else w for w in words]
    return " ".join(words)

def random_deletion(text, p=0.1):
    """EDA: Delete each word with probability p."""
    words = text.split()
    if len(words) == 1:
        return text
    result = [w for w in words if random.random() > p]
    return " ".join(result) if result else random.choice(words)

def random_swap(text, n=1):
    """EDA: Swap n random pairs of words."""
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)

def random_insertion(text, n=1):
    """EDA: Insert a random synonym near a random word."""
    words = text.split()
    for _ in range(n):
        candidates = [w for w in words if get_synonyms(w)]
        if not candidates:
            break
        word = random.choice(candidates)
        syns = get_synonyms(word)
        if syns:
            pos = random.randint(0, len(words))
            words.insert(pos, random.choice(syns))
    return " ".join(words)

def augment_eda(text, num_aug=4, alpha=0.1):
    """Apply all four EDA operations."""
    n = max(1, int(alpha * len(text.split())))
    augmented = []
    for _ in range(num_aug):
        op = random.choice(["sr", "rd", "rs", "ri"])
        if op == "sr":
            augmented.append(synonym_replacement(text, n))
        elif op == "rd":
            augmented.append(random_deletion(text, alpha))
        elif op == "rs":
            augmented.append(random_swap(text, n))
        else:
            augmented.append(random_insertion(text, n))
    return augmented

def augment_dataset(df, num_aug=4, alpha=0.1, seed=42):
    """Return augmented DataFrame (original + augmented rows)."""
    set_seed(seed)
    rows = []
    for _, row in df.iterrows():
        rows.append(row)
        for aug_text in augment_eda(row["clean_text"], num_aug=num_aug, alpha=alpha):
            new_row = row.copy()
            new_row["clean_text"] = aug_text
            rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)

# ─── Full Pipeline ───────────────────────────────────────────────────────────────
def run_pipeline(train_df, val_df, test_df,
                 num_classes, num_epochs=10, batch_size=64,
                 embed_dim=100, lr=1e-3, seed=42, augment=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if augment:
        train_df = augment_dataset(train_df, num_aug=4, alpha=0.1, seed=seed)
        print(f"  Augmented train size: {len(train_df)}")

    vocab = Vocabulary()
    vocab.build(train_df["clean_text"].tolist())

    def make_loader(df, shuffle=False):
        ds = TextDataset(df["clean_text"].tolist(),
                         df["label"].tolist(), vocab)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df)
    test_loader  = make_loader(test_df)

    model     = TextCNN(len(vocab), embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_acc = 0
    best_state   = None

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": loss,
                        "val_acc": val_acc, "val_f1": val_f1})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 2 == 0:
            print(f"    Epoch {epoch:02d} | Loss {loss:.4f} | Val Acc {val_acc:.4f} | F1 {val_f1:.4f}")

    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"  ➜ Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    return test_acc, test_f1, history, model, vocab

if __name__ == "__main__":
    # Quick smoke-test on AG News low-resource splits
    train_df = pd.read_csv("data/agnews_train.csv")
    val_df   = pd.read_csv("data/agnews_val.csv")
    test_df  = pd.read_csv("data/agnews_test.csv")

    num_classes = train_df["label"].nunique()

    print("\n=== Baseline (no augmentation) ===")
    acc, f1, _, _, _ = run_pipeline(train_df, val_df, test_df,
                                    num_classes, num_epochs=10, seed=SEEDS[0])

    print("\n=== With EDA Augmentation ===")
    acc_aug, f1_aug, _, _, _ = run_pipeline(train_df, val_df, test_df,
                                            num_classes, num_epochs=10,
                                            seed=SEEDS[0], augment=True)

    print(f"\nBaseline: Acc={acc:.4f}, F1={f1:.4f}")
    print(f"Augmented: Acc={acc_aug:.4f}, F1={f1_aug:.4f}")
