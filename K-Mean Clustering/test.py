import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from os import path
from sklearn.preprocessing import LabelEncoder

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "plant_growth_data_classification.csv")

# Load dataset
df = pd.read_csv(DATA_FILE)

# Encode categorical features
label_encoders = {}
for column in ["Soil_Type", "Water_Frequency"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store the encoders for reference

# Select features for clustering
features = df[["Sunlight_Hours", "Soil_Type", "Water_Frequency"]].values

# Euclidean distance function
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
    new_centroids = []
    for cluster in clusters:
        if cluster:
            new_centroids.append(np.mean(cluster, axis=0).tolist())
    return new_centroids

# Check if centroids have converged
def has_converged(old_centroids, new_centroids):
    return all(euclidean_distance(old, new) < 1e-6 for old, new in zip(old_centroids, new_centroids))

# K-Means clustering function
def k_means(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters, labels = assign_clusters(data, centroids)
        new_centroids = compute_centroids(clusters)
        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids, clusters

# Compute WCSS for the elbow method
def compute_wcss(data, labels, centroids):
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
        wcss += sum(euclidean_distance(point, centroid) ** 2 for point in cluster_points)
    return wcss

# Compute angle between three points (1D)
def compute_angle(p1, p2, p3):
    # Treat points as 1D (only y-values)
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate the angle using the dot product formula
    dot_product = v1 * v2
    magnitude_v1 = abs(v1)
    magnitude_v2 = abs(v2)

    # Avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 180  # Return a flat angle if one of the vectors is zero

    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    cos_theta = max(min(cos_theta, 1), -1)  # Clamp to avoid numerical errors
    angle = math.acos(cos_theta)
    return math.degrees(angle)


# Elbow Method to find optimal k
def elbow_method(data, max_k=10, threshold=0.5):
    wcss_values = []
    for k in range(1, max_k + 1):
        labels, centroids, _ = k_means(data, k)
        wcss = compute_wcss(data, labels, centroids)
        wcss_values.append(wcss)

    # Print the computed WCSS values
    print("WCSS values for different k values:")
    for k, wcss in enumerate(wcss_values, 1):
        print(f"k={k}: {wcss}")

    # Plot Elbow Graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), wcss_values, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal k")
    plt.show()

    # Compute angles between consecutive points
    angles = []
    for i in range(1, len(wcss_values) - 1):
        p1 = wcss_values[i - 1]
        p2 = wcss_values[i]
        p3 = wcss_values[i + 1]

        angle = compute_angle(p1, p2, p3)
        angles.append(angle)
        

    # Find the optimal k based on the smallest angle
    if angles:
        optimal_k_index = angles.index(min(angles))
        optimal_k = optimal_k_index + 2  # +2 because we start from k=1 and skip the first point
        print(f"Optimal number of clusters (k) determined automatically: {optimal_k}")
    else:
        print("Most Optimal K was Not Found")
        optimal_k = 1  # Default to 1 if no angles are computed

    return wcss_values, optimal_k


# Find the optimal k using the elbow method
wcss_values, optimal_k = elbow_method(features)

# Run k-means with the optimal k
labels, centroids, clusters = k_means(features, optimal_k)

# Save results
df["Cluster"] = labels
output_file = path.join(DATA_DIR, "plant_growth_clustered.csv")
df.to_csv(output_file, index=False)

print(f"Clustering complete. Results saved to {output_file}.")
