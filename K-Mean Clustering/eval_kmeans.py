import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from sklearn.model_selection import train_test_split

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
CLUSTERED_FILE = path.join(DATA_DIR, "wine_clustered.csv")

# Load clustered dataset
df = pd.read_csv(CLUSTERED_FILE)

# Split dataset into 70% training and 30% testing
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Class"])

# Extract actual labels and predicted clusters for test set
true_labels = test_df["Class"].values  # Wine dataset classes: 1, 2, 3
predicted_clusters = test_df["Cluster"].values  # K-Means assigned clusters

# Map clusters to actual labels using majority voting
cluster_to_label = {}
for cluster in np.unique(predicted_clusters):
    cluster_points = train_df[train_df["Cluster"] == cluster]  # Use training set for mapping
    if not cluster_points.empty:
        most_common_label = cluster_points["Class"].mode()[0]  # Most frequent class in this cluster
        cluster_to_label[cluster] = most_common_label

# Assign new predicted labels based on the mapping
mapped_predictions = np.array([cluster_to_label.get(c, -1) for c in predicted_clusters])  # Handle unmapped cases

# Create confusion matrix manually
unique_labels = np.unique(true_labels)  # Ensure we use the original labels (1, 2, 3)
num_classes = len(unique_labels)
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

# Populate the confusion matrix
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}  # Mapping class to index
for true, pred in zip(true_labels, mapped_predictions):
    if pred in label_to_index:  # Ensure valid mapping
        conf_matrix[label_to_index[true], label_to_index[pred]] += 1

# Compute precision, recall, and F1-score manually for each class
precision = []
recall = []
f1_scores = []
for i in range(num_classes):
    TP = conf_matrix[i, i]
    FP = sum(conf_matrix[:, i]) - TP
    FN = sum(conf_matrix[i, :]) - TP
    precision_i = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_i = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0
    precision.append(precision_i)
    recall.append(recall_i)
    f1_scores.append(f1_i)

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
