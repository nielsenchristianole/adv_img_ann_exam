import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb

from scipy.io import loadmat
from sklearn.cluster import KMeans


def plot_points(points):
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))

    for i in range(5):
        ax[i].scatter(points[0, i][:, 0], points[0, i][:, 1], s=2)
        ax[i].grid(False)

    plt.show()


def pairwise_distances(points):
    """
    Calculate pairwise distances between points
    :param points: array of shape (n, 2)
    :return: array of shape (n, n)
    """
    distances = np.linalg.norm(points[:, None] - points, axis=-1)

    return distances


def similarity(distances, sigma):
    """
    Calculate similarity matrix from distances
    :param distances: array of shape (n, n)
    :return: array of shape (n, n)
    """
    return np.exp(-distances ** 2 / (2*sigma**2))


def plot_clusters(point_set, labels):
    colors = np.array(["red", "green", "blue"])
    plt.scatter(point_set[:, 0], point_set[:, 1], c=colors[labels], s=2)
    plt.show()

if __name__ == "__main__":
    snb.set_style("darkgrid")

    points = loadmat("../spectral_data/spectral_data/points_data.mat")["points"]

    plot_points(points)

    # Pairwise distances between all points in the first set
    distances = pairwise_distances(points[0, 0])

    # Similarity matrix of the first set
    similarity_matrix = similarity(distances, 0.5)
    # Threshold the small values of the similarity matrix to zero
    similarity_matrix = (similarity_matrix > 1e-3) * similarity_matrix

    # D (Degree matrix of W)
    degrees = np.count_nonzero(similarity_matrix, axis=1)
    degree_matrix = np.diag(degrees)

    laplacian = degree_matrix - similarity_matrix

    sqrt_degree_matrix = np.diag(1 / np.sqrt(degrees))
    inv_degree_matrix = np.diag(1 / degrees)

    laplacian_sym = sqrt_degree_matrix @ laplacian @ sqrt_degree_matrix
    laplacian_rw = inv_degree_matrix @ laplacian

    evalues, evectors = np.linalg.eigh(laplacian_sym)

    rw_evectors = sqrt_degree_matrix @ evectors

    k = 3

    U = evectors[:, :k]

    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
    labels = kmeans.labels_

    plot_clusters(points[0, 0], labels)



    print("Done")




