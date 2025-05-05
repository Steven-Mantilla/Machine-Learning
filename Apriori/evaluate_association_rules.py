# evaluate_association_rules.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSVs
itemsets = pd.read_csv("frequent_itemsets.csv")
rules = pd.read_csv("association_rules.csv")

# Convert frozenset strings to readable format if needed
def clean_set(val):
    return val.strip("{}").replace(",", "^").replace("'", "")

rules['antecedents'] = rules['antecedents'].apply(clean_set)
rules['consequents'] = rules['consequents'].apply(clean_set)
rules['rule'] = rules['antecedents'] + " => " + rules['consequents']

# Filter strong rules
strong_rules = rules[(rules['confidence'] >= 0.6)]
if 'lift' in rules.columns:
    strong_rules = strong_rules[strong_rules['lift'] > 1]

# Save filtered rules
strong_rules.to_csv("strong_rules.csv", index=False)

# Visualization 1: Top Rules by Confidence
top_conf = rules.sort_values("confidence", ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(data=top_conf, x="confidence", y="rule", palette="Blues_r")
plt.title("Top 10 Rules by Confidence")
plt.xlabel("Confidence")
plt.ylabel("Rule")
plt.tight_layout()
plt.savefig("top_rules_confidence.png")
plt.show()

# Visualization 2: Support vs Confidence
plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=rules,
    x="support",
    y="confidence",
    hue="lift" if 'lift' in rules.columns else None,
    palette="coolwarm",
    size="confidence"
)
plt.title("Support vs Confidence (color = lift)")
plt.tight_layout()
plt.savefig("support_vs_confidence.png")
plt.show()

# Summary report
with open("association_summary.txt", "w") as f:
    f.write("Association Rule Summary\n")
    f.write("=========================\n")
    f.write(f"Total rules: {len(rules)}\n")
    f.write(f"Strong rules (confidence ≥ 0.6): {len(strong_rules)}\n")
    if 'lift' in rules.columns:
        f.write(f"Strong rules with lift > 1: {len(strong_rules)}\n\n")
    f.write("Top 5 Strong Rules:\n")
    for i, row in strong_rules.head(5).iterrows():
        f.write(f"- {row['rule']} (conf={row['confidence']:.2f}, support={row['support']:.2f}, lift={row.get('lift', 0):.2f})\n")

print("Evaluation complete. Files saved:")
print("- strong_rules.csv")
print("- top_rules_confidence.png")
print("- support_vs_confidence.png")
print("- association_summary.txt")
