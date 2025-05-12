# evaluate_association_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load CSV files ===
itemsets = pd.read_csv("frequent_itemsets.csv")
rules = pd.read_csv("association_rules.csv")

# === Helper: Clean frozenset-like string ===
def clean_set(val):
    return val.strip("{}").replace("'", "").replace(",", "^")

# === Clean & prepare ===
itemsets['itemset'] = itemsets['itemset'].apply(clean_set)
rules['antecedents'] = rules['antecedents'].apply(clean_set)
rules['consequents'] = rules['consequents'].apply(clean_set)
rules['rule'] = rules['antecedents'] + " => " + rules['consequents']

# === Evaluate frequent itemsets ===
top_itemsets = itemsets.sort_values("support", ascending=False).head(10)
top_itemsets.to_csv("top_frequent_itemsets.csv", index=False)

# === Evaluate association rules ===
strong_rules = rules[rules['confidence'] >= 0.6]
if 'lift' in rules.columns:
    strong_rules = strong_rules[strong_rules['lift'] > 1]

strong_rules.to_csv("strong_rules.csv", index=False)

# === Plot: Frequent Itemsets ===
plt.figure(figsize=(8, 5))
sns.barplot(data=top_itemsets, x="support", y="itemset", hue="itemset", legend=False, palette="Greens_r")
plt.title("Top 10 Frequent Itemsets by Support")
plt.xlabel("Support")
plt.ylabel("Itemset")
plt.tight_layout()
plt.savefig("top_frequent_itemsets.png")
plt.close()

# === Plot: Top Rules by Confidence ===
top_conf = rules.sort_values("confidence", ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(data=top_conf, x="confidence", y="rule", hue="rule", legend=False, palette="Blues_r")
plt.title("Top 10 Rules by Confidence")
plt.xlabel("Confidence")
plt.ylabel("Rule")
plt.tight_layout()
plt.savefig("top_rules_confidence.png")
plt.close()

# === Plot: Support vs Confidence ===
plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=rules, 
    x="support", 
    y="confidence", 
    hue="lift" if 'lift' in rules.columns else None, 
    palette="coolwarm", 
    size="confidence"
)
plt.title("Support vs Confidence")
plt.tight_layout()
plt.savefig("support_vs_confidence.png")
plt.close()

# === Summary Report ===
with open("association_summary.txt", "w") as f:
    f.write("Association Rule Mining Summary\n")
    f.write("====================================\n\n")
    f.write(f"Total Itemsets: {len(itemsets)}\n")
    f.write(f"Top Frequent Itemsets (support >= {top_itemsets['support'].min():.2f}): {len(top_itemsets)}\n\n")


    f.write(f"Total Rules: {len(rules)}\n")
    f.write(f"Strong Rules (confidence >= 0.5): {len(strong_rules)}\n")
    if 'lift' in rules.columns:
        f.write(f"Strong Rules with lift > 1: {len(strong_rules)}\n\n")
    
    f.write("Top 5 Frequent Itemsets:\n")
    for _, row in top_itemsets.head(5).iterrows():
        f.write(f"- {row['itemset']} (support = {row['support']:.2f})\n")
    
    f.write("\nTop 5 Strong Rules:\n")
    for _, row in strong_rules.head(5).iterrows():
        lift_part = f", lift = {row['lift']:.2f}" if 'lift' in row else ""
        f.write(f"- {row['rule']} (support = {row['support']:.2f}, confidence = {row['confidence']:.2f}{lift_part})\n")

print("Evaluation done. Outputs:")
print("- top_frequent_itemsets.csv")
print("- strong_rules.csv")
print("- top_frequent_itemsets.png")
print("- top_rules_confidence.png")
print("- support_vs_confidence.png")
print("- association_summary.txt")
