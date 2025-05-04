import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import itertools
from os import path
# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "healthcare_dataset.csv")

# 1. Load the CSV file
df = pd.read_csv(DATA_FILE)

# Preprocess: one-hot encode
def encode(df, prefix=True):
    records = df.astype(str).values.tolist()
    if prefix:
        records = [[f"{col}={val}" for col, val in zip(df.columns, row)] for row in records]
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(records).transform(records), columns=te.columns_)

df_enc = encode(df)

# Mine frequent itemsets
min_support = 0.1  # adjust as needed
itemsets = apriori(df_enc, min_support=min_support, use_colnames=True)
itemsets['support'] = itemsets['support'].round(3)

# Step-by-step rule enumeration as in lecture
print("\nRule enumeration (with raw confidence calculations):")
# build a dict for support lookup
support_dict = {frozenset(row['itemsets']): row['support'] for _, row in itemsets.iterrows()}
for itemset in support_dict:
    if len(itemset) < 2:
        continue
    for i in range(1, len(itemset)):
        for antecedent in itertools.combinations(itemset, i):
            antecedent = frozenset(antecedent)
            consequent = itemset - antecedent
            num = support_dict[itemset]
            denom = support_dict.get(antecedent, 0)
            if denom > 0:
                conf = num / denom
                pct = round(conf * 100)
                status = "Selected" if conf >= 0.4 else "Rejected"
                ant_str = "^".join(sorted(antecedent))
                cons_str = "^".join(sorted(consequent))
                print(f"[{ant_str}]=>[{cons_str}] // confidence = {num}/{denom}*100 = {pct}% // {status}")

# After enumeration, produce summary tables (support & confidence only)
from mlxtend.frequent_patterns import association_rules
rules = association_rules(itemsets, metric="confidence", min_threshold=0.4)
rules = rules[['antecedents','consequents','support','confidence']]
rules[['support','confidence']] = rules[['support','confidence']].round(3)

# Sort and select top
top_n = 10
itemsets_top = itemsets.sort_values('support', ascending=False).head(top_n)
rules_top = rules.sort_values('confidence', ascending=False).head(top_n)

# Helper: print markdown table without tabulate

def print_md_table(df, cols):
    # header
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"]*len(cols)) + "|"
    print(header)
    print(sep)
    for _, row in df.iterrows():
        values = []
        for c in cols:
            v = row[c]
            if isinstance(v, frozenset):
                v = "{" + ", ".join(sorted(v)) + "}"
            values.append(str(v))
        print("| " + " | ".join(values) + " |")

# Display summary tables
print("\nTop Frequent Itemsets:")
print_md_table(itemsets_top, ['itemsets','support'])

print("\nTop Association Rules (no lift):")
print_md_table(rules_top, ['antecedents','consequents','support','confidence'])

# Save outputs
itemsets_top.to_csv("frequent_itemsets.csv", index=False)
rules_top.to_csv("association_rules.csv", index=False)
