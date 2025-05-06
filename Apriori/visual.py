import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from os import path

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "healthcare_dataset.csv")

# 1. Load the CSV file
df = pd.read_csv(DATA_FILE)

# 2. Preprocess: one-hot encode
def encode(df, prefix=True):
    records = df.astype(str).values.tolist()
    if prefix:
        records = [[f"{col}={val}" for col, val in zip(df.columns, row)] for row in records]
    unique = sorted({item for row in records for item in row})
    return pd.DataFrame([{item: (item in row) for item in unique} for row in records])

df_enc = encode(df)

# 3. Mine frequent itemsets
min_support = 0.05  # set threshold
itemsets = []
supports = []
for k in range(1, len(df_enc.columns) + 1):
    for combo in combinations(df_enc.columns, k):
        sup = df_enc[list(combo)].all(axis=1).mean()
        if sup >= min_support:
            itemsets.append(combo)
            supports.append(round(sup, 3))
itemsets_df = pd.DataFrame({'itemset': itemsets, 'support': supports})
itemsets_df.to_csv("frequent_itemsets.csv", index=False)

# 4. Step-by-step rule enumeration with clear grouping
confidence_threshold = 0.4
print("\nRule enumeration (grouped by itemset):")
support_lookup = {frozenset(combo): sup for combo, sup in zip(itemsets, supports)}
for combo in itemsets:
    if len(combo) < 2:
        continue
    print(f"\nRules for itemset {set(combo)}:")
    # enumerate rules for this itemset
    for r in range(1, len(combo)):
        for antecedent in combinations(combo, r):
            antecedent = frozenset(antecedent)
            consequent = frozenset(combo) - antecedent
            num = support_lookup[frozenset(combo)]
            denom = support_lookup.get(antecedent, 0)
            if denom > 0:
                conf = num / denom
                pct = round(conf * 100)
                status = 'Selected' if conf >= confidence_threshold else 'Rejected'
                ant_str = '^'.join(sorted(antecedent))
                cons_str = '^'.join(sorted(consequent))
                print(f"  [{ant_str}]=>[{cons_str}]  confidence = {num}/{denom:.3f}*100 = {pct}%  // {status}")

# 5. Summary rules (support & confidence)
rules = []
for combo in itemsets:
    if len(combo) < 2:
        continue
    for r in range(1, len(combo)):
        for antecedent in combinations(combo, r):
            antecedent = frozenset(antecedent)
            consequent = frozenset(combo) - antecedent
            num = support_lookup[frozenset(combo)]
            denom = support_lookup.get(antecedent, 0)
            if denom > 0:
                conf = round(num/denom, 3)
                rules.append({'antecedents': antecedent,
                              'consequents': consequent,
                              'support': support_lookup[frozenset(combo)],
                              'confidence': conf})
rules_df = pd.DataFrame(rules)
rules_df.to_csv("association_rules.csv", index=False)

# 6. Optimized lecture-style visualizations (only Selected itemsets)
min_count = int(min_support * len(df_enc))
k = 1
while True:
    combos_k = [c for c in combinations(df_enc.columns, k)]
    supports_k = [df_enc[list(c)].all(axis=1).mean() for c in combos_k]
    counts_k = [int(s * len(df_enc)) for s in supports_k]
    selected = [(c, s, cnt) for c, s, cnt in zip(combos_k, supports_k, counts_k) if s >= min_support]
    if not selected:
        break

    combos_sel, supports_sel, counts_sel = zip(*selected)
    labels = [' & '.join(c) for c in combos_sel]

    plt.figure(figsize=(8,4))
    plt.bar(range(len(supports_sel)), supports_sel)
    plt.axhline(min_support, linestyle='--')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Support')
    plt.title(f"{k}-itemsets Support (only selected, min count={min_count})")
    plt.tight_layout()
    plt.show()

    print(f"\n{k}-itemsets Selected (count >= {min_count}):")
    for c, s, cnt in selected:
        print(f"  {set(c)} -> count = {cnt} ({s:.3f})")

    k += 1
