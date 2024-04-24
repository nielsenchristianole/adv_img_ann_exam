import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb

from scipy.io import loadmat
from sklearn.cluster import KMeans


def plot_points(points, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 5, figsize=(20, 4))

    for i in range(len(ax)):
        ax[i].scatter(points[0, i][:, 0], points[0, i][:, 1], s=2)



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


def plot_clusters(point_set, labels, ax=None):
    colors = np.array(["red", "green", "blue"])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plt.scatter(point_set[:, 0], point_set[:, 1], c=colors[labels], s=2)
        plt.show()
    else:
        ax.scatter(point_set[:, 0], point_set[:, 1], c=colors[labels], s=2)


if __name__ == "__main__":
    snb.set_style("darkgrid")
    plt.rcParams["axes.grid"] = False

    fig, ax = plt.subplots(4, 5, figsize=(20, 16))

    points = loadmat("../spectral_data/spectral_data/points_data.mat")["points"]

    clusters_per_set = [3, 3, 2, 3, 2]
    sigmas = [0.65, 1.5, 1.25, 1.25, 1.5]

    plot_points(points, ax[0, :])

    # Pairwise distances between all points in the first set
    for i, (k, sigma) in enumerate(zip(clusters_per_set, sigmas)):
        # Compute the pairwise distances between all points for each set
        distances = pairwise_distances(points[0, i])
        # Construct the similarity matrix from the distances
        similarity_matrix = similarity(distances, sigma)
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

        U = rw_evectors[:, 1:k+1]

        kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
        labels = kmeans.labels_

        ax[2, i].matshow(similarity_matrix)
        ax[3, i].matshow(similarity_matrix[labels.argsort()][:, labels.argsort()])

        plot_clusters(points[0, i], labels, ax[1, i])


    plt.show()



    print("Done")




