import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from os import path
from mpl_toolkits.mplot3d import Axes3D

# Set fixed random seeds for reproducibility

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "wine.data")

# Load dataset with column names
column_names = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavonoids", "Nonflavonoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]
df = pd.read_csv(DATA_FILE, header=None, names=column_names)

# Selected features for clustering and analysis
selected_features = ["Alcohol", "Flavonoids", "Malic acid"]
X = df[selected_features].values

# Manual Min-Max Scaling (for clustering)
def min_max_scale(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals), min_vals, max_vals

X_scaled, min_vals, max_vals = min_max_scale(X)

# Euclidean distance function
def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))

# Initialize k random centroids (using random.sample)
def initialize_centroids(data, k):
    return random.sample(data.tolist(), k)

# Assign each point to the nearest centroid
def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    labels = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
        labels.append(cluster_index)
    return clusters, labels

# Compute new centroids
def compute_centroids(clusters):
    # If a cluster is empty, choose a random point from the dataset (scaled)
    return [np.mean(cluster, axis=0).tolist() if cluster else random.choice(X_scaled.tolist()) 
            for cluster in clusters]

# Check if centroids have converged
def has_converged(old_centroids, new_centroids):
    return all(euclidean_distance(old, new) < 1e-6 for old, new in zip(old_centroids, new_centroids))

# Single-run K-Means (returns labels, initial centroids, final centroids, clusters)
def k_means_single_run(data, k):
    initial_centroids = initialize_centroids(data, k)
    centroids = initial_centroids
    max_iters = len(data)
    snapshots = []  # to record each iteration's state
    for it in range(max_iters):
        clusters, labels = assign_clusters(data, centroids)
        # Save a snapshot (make a deep copy of centroids and labels)
        snapshots.append((it, [c.copy() for c in centroids], labels.copy()))
        new_centroids = compute_centroids(clusters)
        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, initial_centroids, centroids, clusters, snapshots

# K-Means clustering with multiple random restarts (selects best run by WCSS)
def k_means(data, k, n_restarts=1):
    best_wcss = float("inf")
    best_result = None
    for _ in range(n_restarts):
        labels, _, centroids, clusters, _ = k_means_single_run(data, k)
        current_wcss = compute_wcss(data, labels, centroids)
        if current_wcss < best_wcss:
            best_wcss = current_wcss
            best_result = (labels, centroids, clusters)
    return best_result

# Compute WCSS for clustering result
def compute_wcss(data, labels, centroids):
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
        wcss += sum(euclidean_distance(point, centroid) ** 2 for point in cluster_points)
    return wcss

# Inverse scaling: convert scaled centroids back to original scale
def inverse_scale(centroids_scaled, min_vals, max_vals):
    return np.array(centroids_scaled) * (max_vals - min_vals) + min_vals

# Compute angle between three points in 2D space (each point is (k, WCSS))
def compute_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 180
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.degrees(np.arccos(cos_theta))

def elbow_method(data, max_k=10):
    wcss_values = []
    k_values = list(range(1, max_k + 1))
    for k in k_values:
        # Use k_means (multiple restarts) for each k
        labels, centroids, clusters = k_means(data, k, n_restarts=1)
        wcss = compute_wcss(data, labels, centroids)
        wcss_values.append(wcss)
    print("WCSS values for different k values:")
    for k, wcss in zip(k_values, wcss_values):
        print(f"k={k}: {wcss}")
    angles = []
    for i in range(1, len(k_values) - 1):
        p1 = (k_values[i-1], wcss_values[i-1])
        p2 = (k_values[i], wcss_values[i])
        p3 = (k_values[i+1], wcss_values[i+1])
        angle = compute_angle(p1, p2, p3)
        angles.append(angle)
    print("Angles computed for each k (at k=2...max_k-1):")
    for i, angle in enumerate(angles):
        print(f"k={i+2}: {angle:.4f} degrees")
    if angles:
        optimal_k_index = np.argmin(angles)
        optimal_k = optimal_k_index + 2  # Because angles start at k=2
    else:
        optimal_k = 3
    print(f"Selected k={optimal_k} with angle={angles[optimal_k-2]:.4f} degrees")
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss_values, marker='o', linestyle='--', label="WCSS")
    plt.scatter(optimal_k, wcss_values[optimal_k - 1], color="red", s=150, label="Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal k")
    plt.legend()
    plt.show()
    return optimal_k

# Find optimal k using the elbow method
optimal_k = elbow_method(X_scaled, max_k=10)

# Obtain initial clustering result (single run) for the initial centroids, labels, and snapshots
initial_labels, initial_centroids_scaled, _, _, snapshots = k_means_single_run(X_scaled, optimal_k)
initial_centroids_original = inverse_scale(initial_centroids_scaled, min_vals, max_vals)

# Run final K-Means with multiple restarts to get final clustering result
labels, final_centroids_scaled, clusters = k_means(X_scaled, optimal_k, n_restarts=5)
final_centroids_original = inverse_scale(final_centroids_scaled, min_vals, max_vals)

# Add cluster assignments to DataFrame and save results
df["Cluster"] = labels
output_file = path.join(DATA_DIR, "wine_clustered.csv")
df.to_csv(output_file, index=False)
print(f"Clustering complete. Results saved to {output_file}.")

# Print cluster details: range (min, max, mean) of selected features and class distribution
def print_cluster_details(df, selected_features):
    for cluster in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == cluster]
        print(f"\nCluster {cluster}:")
        for feat in selected_features:
            feat_min = subset[feat].min()
            feat_max = subset[feat].max()
            feat_mean = subset[feat].mean()
            print(f"  {feat}: min={feat_min:.2f}, max={feat_max:.2f}, mean={feat_mean:.2f}")
        class_distribution = subset["Class"].value_counts().to_dict()
        print("  Class distribution:", class_distribution)

print_cluster_details(df, selected_features)

# Create subplot with two 3D scatter plots:
# Left: Data points colored by initial clustering assignments (from the initial centroids)
# Right: Final clustering result with final centroids
fig = plt.figure(figsize=(16, 7))

# Left subplot: Initial clustering result
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(df["Alcohol"], df["Flavonoids"], df["Malic acid"], 
                       c=initial_labels, cmap="viridis", s=50, label="Data Points (Initial)")
ax1.scatter(initial_centroids_original[:, 0], initial_centroids_original[:, 1], 
            initial_centroids_original[:, 2], c="red", marker="X", s=200, edgecolors="black", 
            label="Initial Centroids")
ax1.set_xlabel("Alcohol")
ax1.set_ylabel("Flavonoids")
ax1.set_zlabel("Malic acid")
ax1.set_title("Before Clustering (Initial Assignment)")
ax1.legend()

# Right subplot: Final clustering result
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(df["Alcohol"], df["Flavonoids"], df["Malic acid"], 
                       c=df["Cluster"], cmap="viridis", s=50, label="Data Points (Final)")
ax2.scatter(final_centroids_original[:, 0], final_centroids_original[:, 1], 
            final_centroids_original[:, 2], c="red", marker="X", s=200, edgecolors="black", 
            label="Final Centroids")
ax2.set_xlabel("Alcohol")
ax2.set_ylabel("Flavonoids")
ax2.set_zlabel("Malic acid")
ax2.set_title(f"Final Clustering (k={optimal_k})")
ax2.legend()

plt.tight_layout()
plt.show()

# Prompt the user to view the iterative clustering process
user_input = input("Do you want to see the iterative clustering process? (yes/no): ")

if user_input.strip().lower().startswith("y"):
    # Use the snapshots from the single run obtained earlier
    num_iterations = len(snapshots)
    # Decide grid layout: 2 columns, rows as needed.
    ncols = 3
    nrows = math.ceil(num_iterations / ncols)
    
    fig_iter, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), subplot_kw={'projection': '3d'})
    axes = axes.flatten()  # Flatten if more than one row
    
    # For each snapshot, plot data (in original scale) with the centroids from that iteration
    for idx, (it, centroids_snapshot, labels_snapshot) in enumerate(snapshots):
        ax = axes[idx]
        # Plot data points with iterative labels using the same style as final clustering
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                             c=labels_snapshot, cmap="viridis", s=50, label="Data Points (Iterative)")
        # Convert centroids to original scale for visualization
        centroids_snapshot_orig = inverse_scale(centroids_snapshot, min_vals, max_vals)
        ax.scatter(centroids_snapshot_orig[:, 0], centroids_snapshot_orig[:, 1],
                   centroids_snapshot_orig[:, 2], c="red", marker="X", s=200, edgecolors="black",
                   label="Centroids (Iterative)")
        ax.set_xlabel("Alcohol")
        ax.set_ylabel("Flavonoids")
        ax.set_zlabel("Malic acid")
        ax.set_title(f"Iteration {it}")
        ax.legend()
    
    # Hide any extra subplots if there are fewer snapshots than grid cells
    for j in range(idx + 1, len(axes)):
        fig_iter.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()
