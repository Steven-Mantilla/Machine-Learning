import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from os import path

# === Define Paths ===
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "healthcare_dataset.csv")

# === 1. Load the CSV file ===
df = pd.read_csv(DATA_FILE)

# === 2. Preprocess: one-hot encode ===
def encode(df, prefix=True):
    records = df.astype(str).values.tolist()
    if prefix:
        records = [[f"{col}={val}" for col, val in zip(df.columns, row)] for row in records]
    unique = sorted({item for row in records for item in row})
    return pd.DataFrame([{item: (item in row) for item in unique} for row in records])

df_enc = encode(df)

# === 3. Visualize Frequent Itemsets for k = 1, 2, 3 ===
min_support = 0.05  # Set support threshold

for k in [1, 2, 3]:
    itemsets = []
    supports = []
    for combo in combinations(df_enc.columns, k):
        sup = df_enc[list(combo)].all(axis=1).mean()
        if sup >= min_support:
            itemsets.append(combo)
            supports.append(round(sup, 3))

    if not itemsets:
        print(f"No {k}-itemsets found above min support.")
        continue

    # Sort itemsets by support (descending)
    sorted_pairs = sorted(zip(itemsets, supports), key=lambda x: x[1], reverse=True)
    sorted_itemsets, sorted_supports = zip(*sorted_pairs)
    labels = [' & '.join(c) for c in sorted_itemsets]

    plt.figure(figsize=(12, 5))
    plt.bar(labels, sorted_supports, color="skyblue")
    plt.axhline(min_support, linestyle='--', color='red', linewidth=1, label=f"Min Support ({min_support})")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Support")
    plt.title(f"{k}-Itemsets (Support ≥ {min_support}) — Sorted")
    plt.legend()
    plt.tight_layout()
    plt.show()
