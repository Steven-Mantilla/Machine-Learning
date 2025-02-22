import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from os import path

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "wine.data")

# Load dataset
column_names = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]
df = pd.read_csv(DATA_FILE, header=None, names=column_names)

# Select features
selected_features = ["Alcohol", "Flavanoids", "Malic acid"]
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

# Initialize k random centroids
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
    return [np.mean(cluster, axis=0).tolist() if cluster else random.choice(clusters[0]) for cluster in clusters]

# Check if centroids have converged
def has_converged(old_centroids, new_centroids):
    return all(euclidean_distance(old, new) < 1e-6 for old, new in zip(old_centroids, new_centroids))

# K-Means clustering function
def k_means(data, k):
    centroids = initialize_centroids(data, k)
    max_iters = len(data)  # Set iterations to dataset size
    for _ in range(max_iters):
        clusters, labels = assign_clusters(data, centroids)
        new_centroids = compute_centroids(clusters)
        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids, clusters

# Compute WCSS for Elbow Method
def compute_wcss(data, labels, centroids):
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
        wcss += sum(euclidean_distance(point, centroid) ** 2 for point in cluster_points)
    return wcss

# Compute angle for Elbow Method
def compute_angle(p1, p2, p3):
    """
    Computes the angle between three points in 2D space.
    The points represent (k, WCSS) values for the Elbow Method.
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    # Compute cosine similarity
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 180  # Avoid division by zero
    
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1, 1)  # Ensure numerical stability
    return np.degrees(np.arccos(cos_theta))

def elbow_method(data, max_k=10):
    """
    Implements the Elbow Method with the improved angle calculation.
    Selects the k where the WCSS curve bends the most.
    """
    wcss_values = []
    k_values = list(range(1, max_k + 1))

    for k in k_values:
        labels, centroids, _ = k_means(data, k)
        wcss_values.append(compute_wcss(data, labels, centroids))
        
    # Print the computed WCSS values
    print("WCSS values for different k values:")
    for k, wcss in enumerate(wcss_values, 1):
        print(f"k={k}: {wcss}")
        
    # Compute angles between consecutive points
    angles = []
    for i in range(1, len(k_values) - 1):
        p1 = (k_values[i - 1], wcss_values[i - 1])
        p2 = (k_values[i], wcss_values[i])
        p3 = (k_values[i + 1], wcss_values[i + 1])
        angles.append(compute_angle(p1, p2, p3))
    print("Angles computed for each k:")
    for i, angle in enumerate(angles):
        print(f"k={i+2}: {angle:.4f} degrees")


    # Optimal k is the one with the largest angle (sharpest bend)
    optimal_k_index = np.argmin(angles)  # Find the index of the smallest angle
    optimal_k = optimal_k_index + 2  # Since angles start from k=2
    print(f"Selected k={optimal_k} with angle={angles[optimal_k_index]:.4f}")
        

    # Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss_values, marker='o', linestyle='--', label="WCSS")
    plt.scatter(optimal_k, wcss_values[optimal_k - 1], color="red", s=150, label="Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal k")
    plt.legend()
    plt.show()

    return optimal_k

# Find optimal k
optimal_k = elbow_method(X_scaled)

# Run K-Means
labels, centroids_scaled, clusters = k_means(X_scaled, optimal_k)

# Convert centroids back to original scale
centroids_original = np.array(centroids_scaled) * (max_vals - min_vals) + min_vals

# Add clusters to DataFrame
df["Cluster"] = labels

# Save results
output_file = path.join(DATA_DIR, "wine_clustered.csv")
df.to_csv(output_file, index=False)
print(f"Clustering complete. Results saved to {output_file}.")

# 3D Scatter Plot with Centroids
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot Data Points
scatter = ax.scatter(
    df["Alcohol"], df["Flavanoids"], df["Malic acid"], 
    c=df["Cluster"], cmap="viridis", s=50, label="Data Points"
)

# Plot Centroids
ax.scatter(
    centroids_original[:, 0], centroids_original[:, 1], centroids_original[:, 2],
    c="red", marker="X", s=200, edgecolors="black", label="Centroids"
)

# Labels & Legend
ax.set_xlabel("Alcohol")
ax.set_ylabel("Flavanoids")
ax.set_zlabel("Malic Acid")
plt.colorbar(scatter, label="Cluster")
plt.legend()
plt.title(f"3D Scatter Plot of Clusters with Centroids (k={optimal_k})")
plt.show()
