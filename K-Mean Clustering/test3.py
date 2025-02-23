import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import path

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "Mall_Customers_Preprocessed.csv")

# Load preprocessed dataset
df = pd.read_csv(DATA_FILE)

# Select features
selected_features = ["Annual Income (k$)", "Spending Score (1-100)", "Age"]
X = df[selected_features].values

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
    max_iters = 100  # Set maximum iterations
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

# Elbow Method to find optimal k
def elbow_method(data, max_k=10):
    wcss_values = []
    k_values = list(range(1, max_k + 1))

    for k in k_values:
        labels, centroids, _ = k_means(data, k)
        wcss_values.append(compute_wcss(data, labels, centroids))
    
    print("WCSS values for different k values:")
    for k, wcss in enumerate(wcss_values, 1):
        print(f"k={k}: {wcss}")
    
    angles = []
    for i in range(1, len(k_values) - 1):
        p1 = (k_values[i - 1], wcss_values[i - 1])
        p2 = (k_values[i], wcss_values[i])
        p3 = (k_values[i + 1], wcss_values[i + 1])
        angles.append(compute_angle(p1, p2, p3))
    
    print("Angles computed for each k:")
    for i, angle in enumerate(angles):
        print(f"k={i+2}: {angle:.4f} degrees")
    
    optimal_k_index = np.argmin(angles)
    optimal_k = optimal_k_index + 2
    print(f"Selected k={optimal_k} with angle={angles[optimal_k_index]:.4f}")
    
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
optimal_k = elbow_method(X)

# Run K-Means with optimal k
labels, centroids, clusters = k_means(X, optimal_k)

df["Cluster"] = labels

output_file = path.join(DATA_DIR, "Mall_Customers_Clustered.csv")
df.to_csv(output_file, index=False)
print(f"Clustering complete. Results saved to {output_file}.")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], df["Age"], c=df["Cluster"], cmap="viridis", s=50)
centroids = np.array(centroids)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="red", marker="X", s=200, edgecolors="black", label="Centroids")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_zlabel("Age")
plt.colorbar(scatter, label="Cluster")
plt.legend()
plt.title(f"3D Scatter Plot of Mall Customers Clusters (k={optimal_k})")
plt.show()