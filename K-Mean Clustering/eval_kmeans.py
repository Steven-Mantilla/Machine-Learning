import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load clustered dataset
df = pd.read_csv("Datasets/wine_clustered.csv")

# Split dataset into 70% training and 30% testing
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Class"])

# Extract actual labels and predicted clusters for test set
true_labels = test_df["Class"].values
predicted_clusters = test_df["Cluster"].values

# Map clusters to actual labels using majority voting
cluster_to_label = {
    cluster: train_df[train_df["Cluster"] == cluster]["Class"].mode()[0]
    for cluster in np.unique(predicted_clusters)
    if not train_df[train_df["Cluster"] == cluster].empty
}

# Assign new predicted labels based on the mapping
mapped_predictions = np.array([cluster_to_label.get(c, -1) for c in predicted_clusters])

# Create confusion matrix manually
unique_labels = np.unique(true_labels)
num_classes = len(unique_labels)
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

for true, pred in zip(true_labels, mapped_predictions):
    if pred in label_to_index:
        conf_matrix[label_to_index[true], label_to_index[pred]] += 1

# Compute precision, recall, and F1-score manually
precision, recall, f1_scores = [], [], []
for i in range(num_classes):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP
    precision.append(TP / (TP + FP) if TP + FP > 0 else 0)
    recall.append(TP / (TP + FN) if TP + FN > 0 else 0)
    f1_scores.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1]) if precision[-1] + recall[-1] > 0 else 0)

# Display results
print("\nConfusion Matrix:")
print(conf_matrix)

for i, label in enumerate(unique_labels):
    print(f"\nClass {label}:")
    print(f"Precision: {precision[i]:.2f}")
    print(f"Recall: {recall[i]:.2f}")
    print(f"F1-score: {f1_scores[i]:.2f}")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.xticks(range(num_classes), labels=unique_labels)
plt.yticks(range(num_classes), labels=unique_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix (Test Set)")
plt.show()
