import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from os import path
from collections import Counter

# Set fixed random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "wine.data")

# Load dataset
column_names = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavonoids", "Nonflavonoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]
df = pd.read_csv(DATA_FILE, header=None, names=column_names)

# Select features
selected_features = ["Alcohol", "Flavonoids", "Malic acid"]
X = df[selected_features].values

# Manual Min-Max Scaling
def min_max_scale(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals), min_vals, max_vals

X_scaled, min_vals, max_vals = min_max_scale(X)

# Euclidean Distance Function
def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))

# Initialize k random centroids (fixed seed makes this reproducible)
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
    # If a cluster is empty, choose a random point from the entire dataset
    return [np.mean(cluster, axis=0).tolist() if cluster else random.choice(X_scaled.tolist()) 
            for cluster in clusters]

# Check if centroids have converged
def has_converged(old_centroids, new_centroids):
    return all(euclidean_distance(old, new) < 1e-6 for old, new in zip(old_centroids, new_centroids))

# K-Means clustering function with multiple random restarts
def k_means(data, k, n_restarts=5):
    best_wcss = float("inf")
    best_result = None
    for _ in range(n_restarts):
        centroids = initialize_centroids(data, k)
        max_iters = len(data)
        for _ in range(max_iters):
            clusters, labels = assign_clusters(data, centroids)
            new_centroids = compute_centroids(clusters)
            if has_converged(centroids, new_centroids):
                break
            centroids = new_centroids
        current_wcss = compute_wcss(data, labels, centroids)
        if current_wcss < best_wcss:
            best_wcss = current_wcss
            best_result = (labels, centroids, clusters)
    return best_result

# Compute WCSS for Elbow Method
def compute_wcss(data, labels, centroids):
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
        wcss += sum(euclidean_distance(point, centroid) ** 2 for point in cluster_points)
    return wcss

# Compute angle between three points in 2D space (using (k, WCSS))
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
    
    # For each k, run k-means with multiple restarts
    for k in k_values:
        labels, centroids, _ = k_means(data, k, n_restarts=5)
        wcss_values.append(compute_wcss(data, labels, centroids))
        
    # Print WCSS values for debugging
    print("WCSS values for different k values:")
    for k, wcss in zip(k_values, wcss_values):
        print(f"k={k}: {wcss}")
        
    # Compute angles between consecutive (k, WCSS) points
    angles = []
    for i in range(1, len(k_values) - 1):
        p1 = (k_values[i - 1], wcss_values[i - 1])
        p2 = (k_values[i], wcss_values[i])
        p3 = (k_values[i + 1], wcss_values[i + 1])
        angle = compute_angle(p1, p2, p3)
        angles.append(angle)
    print("Angles computed for each k:")
    for i, angle in enumerate(angles):
        print(f"k={i+2}: {angle:.4f} degrees")
    
    # Select the optimal k (smallest angle indicates a sharper bend)
    if angles:
        optimal_k_index = np.argmin(angles)
        optimal_k = optimal_k_index + 2  # Because angles start at k=2
    else:
        optimal_k = 3  # Fallback
    print(f"Selected k={optimal_k} with angle={angles[optimal_k-2]:.4f} degrees")
    
    plot_elbow(k_values, wcss_values, optimal_k)

    return optimal_k

def plot_elbow(k_values, wcss_values, optimal_k):
    # Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss_values, marker='o', linestyle='--', label="WCSS")
    plt.scatter(optimal_k, wcss_values[optimal_k - 1], color="red", s=150, label="Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal k")
    plt.legend()
    plt.show()

# Assuming elbow_method(X_scaled) returns the optimal k value
def get_optimal_k():
    # Placeholder for the actual implementation of elbow_method
    return elbow_method(X_scaled)


optimal_k = get_optimal_k()
# most_prevalent_k = []

# # Run the process multiple times
# for _ in range(100):
#     optimal_k = get_optimal_k()
#     most_prevalent_k.append(optimal_k)

# # Count the occurrences of each k value
# k_counter = Counter(most_prevalent_k)

# # Find the most prevalent k value
# most_common_k = k_counter.most_common(1)[0]

# # Print the result
# print(f"The most prevalent k value is {most_common_k[0]} with {most_common_k[1]} occurrences.")



# Run K-Means with the optimal k
labels, centroids_scaled, clusters = k_means(X_scaled, optimal_k, n_restarts=5)

# Convert centroids back to original scale
centroids_original = np.array(centroids_scaled) * (max_vals - min_vals) + min_vals

# Add clusters to DataFrame and save results
df["Cluster"] = labels
output_file = path.join(DATA_DIR, "wine_clustered.csv")
df.to_csv(output_file, index=False)
print(f"Clustering complete. Results saved to {output_file}.")

# 3D Scatter Plot with Centroids
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df["Alcohol"], df["Flavonoids"], df["Malic acid"], 
                     c=df["Cluster"], cmap="viridis", s=50, label="Data Points")
ax.scatter(centroids_original[:, 0], centroids_original[:, 1], centroids_original[:, 2],
           c="red", marker="X", s=200, edgecolors="black", label="Centroids")
ax.set_xlabel("Alcohol")
ax.set_ylabel("Flavonoids")
ax.set_zlabel("Malic acid")
plt.colorbar(scatter, label="Cluster")
plt.legend()
plt.title(f"3D Scatter Plot of Clusters with Centroids (k={optimal_k})")
plt.show()
