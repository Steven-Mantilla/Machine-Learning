import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "wine.data")

# Load dataset with column names
column_names = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]
df = pd.read_csv(DATA_FILE, header=None, names=column_names)

# Select features
selected_features = ["Alcohol", "Flavanoids", "Malic acid"]
X = df[selected_features].values

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal k using the elbow method with KneeLocator
inertia_values = []
max_k = 10
k_values = range(1, max_k + 1)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Print inertia (WCSS) values for each k
print("Inertia values for different k:")
for k, inertia in zip(k_values, inertia_values):
    print(f"k={k}: {inertia}")

# Use KneeLocator to detect the elbow
kl = KneeLocator(k_values, inertia_values, curve="convex", direction="decreasing")
optimal_k = kl.elbow
print(f"\nOptimal number of clusters (k) determined by KneeLocator: {optimal_k}")

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o', linestyle='--', label="Inertia")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Optimal k")
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f"Optimal k = {optimal_k}")
plt.legend()
plt.show()

# Run final K-Means clustering with optimal k
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans_final.fit_predict(X_scaled)
centroids_scaled = kmeans_final.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# Save the clustering results
output_file = path.join(DATA_DIR, "wine_clustered.csv")
df.to_csv(output_file, index=False)
print(f"\nClustering complete. Results saved to {output_file}.")

# 3D Scatter Plot with Centroids
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df["Alcohol"], df["Flavanoids"], df["Malic acid"],
                     c=df["Cluster"], cmap="viridis", s=50, label="Data Points")
ax.scatter(centroids_original[:, 0], centroids_original[:, 1], centroids_original[:, 2],
           c="red", marker="X", s=200, edgecolors="black", label="Centroids")
ax.set_xlabel("Alcohol")
ax.set_ylabel("Flavanoids")
ax.set_zlabel("Malic acid")
plt.colorbar(scatter, label="Cluster")
plt.title(f"3D Scatter Plot of Clusters (k={optimal_k})")
plt.legend()
plt.show()
