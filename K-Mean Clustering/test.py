import pandas as pd
import random
import numpy as np
from os import path
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "wine.data")

# Load dataset
column_names = [
    "Class",  # Label
    "Alcohol",  # Feature 1
    "Malic acid",  # Feature 2
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",  # Feature 3
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",  # Feature 4
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
]
df = pd.read_csv(DATA_FILE, header=None, names=column_names)

# Select the first 100 rows using .loc
df_first_100 = df.loc[:99].copy()

# Select only the required features
selected_features = ["Alcohol", "Flavanoids", "Malic acid"]
X_first_100 = df_first_100[selected_features]

# Normalize the features
scaler = StandardScaler()
X_normalized_first_100 = scaler.fit_transform(X_first_100)
X_normalized_first_100_df = pd.DataFrame(X_normalized_first_100, columns=selected_features)

# Combine normalized features into a single dataset
processed_df_first_100 = X_normalized_first_100_df

# Select features for clustering
features_first_100 = processed_df_first_100.values

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

    # Handle potential floating-point precision errors
    if cos_theta < -1:
        cos_theta = -1
    elif cos_theta > 1:
        cos_theta = 1

    # Compute the raw angle
    angle = math.acos(cos_theta)
    return math.degrees(angle)

# Elbow Method to find optimal k
def elbow_method(data, max_k=10):
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

        # Debug: Print the points being used for angle calculation
        print(f"Points for angle calculation: p1={p1}, p2={p2}, p3={p3}")

        angle = compute_angle(p1, p2, p3)
        angles.append(angle)

        # Debug: Print the computed angle
        print(f"Angle for k={i + 1}: {angle} degrees")

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
wcss_values, optimal_k = elbow_method(features_first_100)

# Run k-means with the optimal k
labels_first_100, centroids_first_100, clusters_first_100 = k_means(features_first_100, optimal_k)

# Add the 'Cluster' column using .loc
df_first_100.loc[:, "Cluster"] = labels_first_100

# Save results
output_file_first_100 = path.join(DATA_DIR, "wine_clustered_first_100.csv")
df_first_100.to_csv(output_file_first_100, index=False)

print(f"Clustering complete. Results saved to {output_file_first_100}.")

# Visualize the clusters in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each cluster
scatter = ax.scatter(
    df_first_100["Alcohol"],  # X-axis: Alcohol
    df_first_100["Flavanoids"],  # Y-axis: Flavanoids
    df_first_100["Malic acid"],  # Z-axis: Malic Acid
    c=df_first_100["Cluster"],  # Color by cluster
    cmap="viridis",  # Color map
    s=50,  # Marker size
    depthshade=True  # Add depth shading
)

# Add labels and title
ax.set_xlabel("Alcohol")
ax.set_ylabel("Flavanoids")
ax.set_zlabel("Malic Acid")
plt.colorbar(scatter, label="Cluster")
plt.title(f"3D Scatter Plot of Clusters (k={optimal_k}, First 100 Rows)")

# Show the plot
plt.show()