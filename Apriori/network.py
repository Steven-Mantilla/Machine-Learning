import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os import path

# ==== 1. Paths & Parameters ====
RULES_FILE = "association_rules.csv"
TOP_N      = 15
SPRING_K   = 0.8
FIGSIZE    = (10, 7)
SEPARATOR  = " | "

# ==== 2. Load & Clean ====
rules = pd.read_csv(RULES_FILE)

def clean_set(val: str) -> str:
    """
    Remove the 'frozenset(...)' wrapper and convert set-style strings
    into pipe-delimited labels.
    """
    # 1) Remove frozenset(...) wrapper
    inner = re.sub(r"^frozenset\((.*)\)$", r"\1", val.strip())
    # 2) Strip braces and quotes, then replace commas with SEPARATOR
    cleaned = (
        inner
        .strip("{}")
        .replace("'", "")
        .strip()
    )
    # Replace commas separating items with the pipe separator
    return SEPARATOR.join(item.strip() for item in cleaned.split(","))

# Apply cleaning to antecedents and consequents
rules['antecedents'] = rules['antecedents'].astype(str).apply(clean_set)
rules['consequents']  = rules['consequents'].astype(str).apply(clean_set)

# ==== 3. Filter Top Rules by Confidence ====
top_rules = (
    rules
    .sort_values('confidence', ascending=False)
    .head(TOP_N)
    .reset_index(drop=True)
)

# ==== 4. Build Directed Graph ====
G = nx.DiGraph()
for _, row in top_rules.iterrows():
    a = row['antecedents']
    c = row['consequents']
    w = row['confidence']
    G.add_edge(a, c, weight=w)

# ==== 5. Layout & Edge Coloring ====
pos     = nx.spring_layout(G, k=SPRING_K, seed=42)
edges   = list(G.edges())
weights = [G[u][v]['weight'] for u, v in edges]

norm        = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
edge_colors = [plt.cm.Blues(norm(w)) for w in weights]

# ==== 6. Plot on Explicit Axes ====
fig, ax = plt.subplots(figsize=FIGSIZE)

# Draw nodes
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_size=1800,
    node_color='skyblue',
    alpha=0.9
)

# Draw edges
nx.draw_networkx_edges(
    G, pos, ax=ax,
    edgelist=edges,
    edge_color=edge_colors,
    width=2,
    arrowsize=15
)

# Draw labels
nx.draw_networkx_labels(
    G, pos, ax=ax,
    font_size=10,
    font_color='black'
)

# Add colorbar for confidence
sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Confidence')

ax.set_title("Top Association Rules Network")
ax.axis('off')

plt.tight_layout()
plt.savefig("association_rules_network.png", dpi=300)
plt.show()
