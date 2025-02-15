import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class KmeansCluster:
    def __init__(self, n_clusters, max_iter, random_state):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None



    def intitialize_centroids(self, x_list):
        np.random.seed(self.random_state)
        random_indices = np.random.choice(x_list.shape[0], size=self.n_clusters, replace=False)
        self.centroids = x_list[random_indices]
        return x_list

    def compute_distance(self, x_list, centroids):
        distances = np.zeros(x_list.shape[0], self.n_clusters)

        for i, centroid in enumerate(centroids):
            distances[:, i] = np.linalg.norm(x_list - centroid, axis=1)
        return distances

    def assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def update_centroids(self, x_list, labels):
        centroids = np.zeros((self.n_clusters, x_list.shape[1]))

        for i in range(self.n_clusters):
            centroids[i] = np.mean(x_list[labels ==i], axis=0)

        return centroids

    def fit(self, x_list):
        self.centroids = self.intitialize_centroids(x_list)
        for _ in range(self.max_iter):
            distances = self.compute_distance(x_list, self.centroids)

            self.labels = self.assign_clusters(distances)

            new_centroids = self.update_centroids(x_list, self.labels)


            if np.allclose(new_centroids == self.centroids):
                break
            self.centroids = new_centroids


    def predict(self, x_list):
        distances = self.compute_distance(x_list, self.centroids)
        return self.assign_clusters(distances)