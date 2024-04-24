from typing import Literal, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pylanczos import PyLanczos


IMAGE_NAME = 'peppers'


params = dict(
    plane = dict(
        img_downscale = 6,
        radius = float('inf'),
        sigma_i = 0.5,
        sigma_x = 8,
        k = 2,
        mode='gray'
    ),
    peppers = dict(
        img_downscale = 6,
        radius = float('inf'),
        sigma_i = 0.8,
        sigma_x = 9,
        k = 5,
        mode='rgb'
    )
)


class ImageSegmenter:

    def __init__(
        self,
        *,
        eigen_solver: Literal['full', 'lanczos'] = 'lanczos',
        laplacian: Literal['unnorm', 'shi_malik', 'ng_jordan_weiss'] = 'shi_malik',
        node_size: Literal['volume', 'degree'] = 'volume',
    ) -> None:
        self.eigen_solver = eigen_solver
        self.laplacian = laplacian
        self.node_size = node_size
    
    def __call__(self, similarity_matrix: np.ndarray, num_clusters: int, *, return_encodings: bool=False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        encodings = self.encode_nodes(similarity_matrix, num_clusters)
        labels = self._cluster_encodings(encodings, num_clusters)
        if return_encodings:
            return labels, encodings
        return labels

    def _cluster_encodings(self, encodings: np.ndarray, num_clusters: int) -> np.ndarray:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(encodings)
        return kmeans.labels_

    def _get_node_size(self, similarity_matrix: np.ndarray) -> np.ndarray:
        match self.node_size:
            case 'volume':
                return np.sum(similarity_matrix, axis=1)
            case 'degree':
                return np.count_nonzero(similarity_matrix, axis=1)
    
    def _get_eigen_decomposition(self, similarity_matrix: np.ndarray, num_clusters: int) -> np.ndarray:
        match self.eigen_solver:
            case 'full':
                _, eig_vec =  np.linalg.eigh(similarity_matrix)
            case 'lanczos':
                solver = PyLanczos(similarity_matrix, find_maximum=False, num_eigs=num_clusters+1)
                _, eig_vec = solver.run()
        return eig_vec[:, 1:1+num_clusters]

    def encode_nodes(self, similarity_matrix: np.ndarray, num_clusters: int) -> np.ndarray:
        d = self._get_node_size(similarity_matrix)
        D = np.diag(d)
        D_sqrt_inv = np.diag(1 / np.sqrt(d))

        match self.laplacian:
            case 'unnorm':
                L = D - similarity_matrix
                return self._get_eigen_decomposition(L, num_clusters)
            case 'shi_malik':
                L = D - similarity_matrix
                L_sym = D_sqrt_inv @ L @ D_sqrt_inv
                eig_vec = self._get_eigen_decomposition(L_sym, num_clusters)
                return D_sqrt_inv @ eig_vec
            case 'ng_jordan_weiss':
                L = D - similarity_matrix
                L_sym = D_sqrt_inv @ L @ D_sqrt_inv
                eig_vec = self._get_eigen_decomposition(L_sym, num_clusters)
                return eig_vec / np.linalg.norm(eig_vec, axis=1, keepdims=True)

cluster = ImageSegmenter(
    eigen_solver = 'full',
    laplacian = 'shi_malik',
    node_size = 'volume'
)


img_downscale = params[IMAGE_NAME]['img_downscale']
radius = params[IMAGE_NAME]['radius']
sigma_i = params[IMAGE_NAME]['sigma_i']
sigma_x = params[IMAGE_NAME]['sigma_x']
k = params[IMAGE_NAME]['k']
mode = params[IMAGE_NAME]['mode']


# load image
img_raw = cv2.imread(f'spectral_data/spectral_data/{IMAGE_NAME}.png')
img_raw = cv2.resize(img_raw, (img_raw.shape[1]//img_downscale, img_raw.shape[0]//img_downscale))

img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY).astype(float) / 255
img_col = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB).astype(float) / 255

if mode == 'gray':
    img = img_gray
elif mode == 'rgb':
    img = img_col
elif mode == 'hsv':
    # remove the value channel
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)[:,:,:2].astype(float) / 255

# get params
before_shape = img_gray.shape
mat_lenth = np.prod(before_shape)
if mode == 'gray':
    img = img.reshape(-1)[..., None]
elif mode == 'hsv':
    img = img.reshape(-1, 2)
else:
    img = img.reshape(-1, 3)

# create the coordinates of the image
xs = np.arange(before_shape[1], dtype=float)
ys = np.arange(before_shape[0], dtype=float)
X, Y = np.meshgrid(xs, ys)
X, Y = X.reshape(-1), Y.reshape(-1)

# create the weight matrix
idx = np.arange(mat_lenth, dtype=int)
row_inx, col_idx = np.meshgrid(idx, idx)
row_inx, col_idx = row_inx.reshape(-1), col_idx.reshape(-1)

# get the pixel distances
dist = np.sqrt((X[row_inx] - X[col_idx])**2 + (Y[row_inx] - Y[col_idx])**2)
mask = dist <= radius

# get the intensity differences
int_diffs = np.linalg.norm(img[row_inx] - img[col_idx], axis=1)

# calculate the intensity, spatial and joint weights
w_i = (np.exp(-int_diffs / sigma_i**2)).reshape(mat_lenth, mat_lenth)
w_x = (np.exp(-dist / sigma_x**2) * mask).reshape(mat_lenth, mat_lenth)

w_ij = np.exp((-int_diffs / sigma_i**2) + (-dist / sigma_x**2)) * mask
w_ij = w_ij.reshape(mat_lenth, mat_lenth)

# visualize the weights matrices
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(w_i, cmap='gray')
axs[1].imshow(w_x, cmap='gray')
axs[2].imshow(w_ij, cmap='gray')
axs[0].set_title('Intensity')
axs[1].set_title('Spatial')
axs[2].set_title('Joint')
for ax in axs:
    ax.axis('off')
fig.tight_layout()
fig.savefig(f'results/{IMAGE_NAME}_affinity_matrix.pdf', bbox_inches='tight', pad_inches=0)
# plt.show()

colors = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
], dtype=float)


# plot segmentations
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

metrices = [w_ij, w_i, w_x]
for row, (similarity_matrix, _axs) in enumerate(zip(metrices, axs)):
    labels, _encodings = cluster(similarity_matrix, k, return_encodings=True)
    segmented_img = labels.reshape(before_shape)
    colored_segmentations = colors[segmented_img]
    colored_image = colored_segmentations * img_gray[:, :, None]

    if mode == 'gray':
        _axs[0].imshow(img_gray, cmap='gray')
    elif mode == 'rgb':
        _axs[0].imshow(img_col)
    elif mode == 'hsv':
        _axs[0].imshow(img_col)
    else:
        raise NotImplementedError()
    _axs[1].imshow(colored_image)
    _axs[2].imshow(colored_segmentations)

    if row == 0:
        _axs[0].set_xlabel('Original')
        _axs[1].set_xlabel('Colored by cluster')
        _axs[2].set_xlabel('Segmentations')
        encodings = _encodings
    
    _axs[0].set_ylabel(('Joint', 'Intensity', 'Spatial')[row])

for ax in axs.flatten():
    # ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_label_position('top')


fig.tight_layout()
fig.savefig(f'results/{IMAGE_NAME}_segmented.pdf', bbox_inches='tight', pad_inches=0)
# plt.show()

ordinal_indicator = {1: 'st', 2: 'nd', 3: 'rd'}
fig, axs = plt.subplots(1, k, figsize=(5*k, 5))
for idx, (ax, encoding) in enumerate(zip(axs.flatten(), encodings.T), start=2):
    ax.imshow(encoding.reshape(before_shape), cmap='gray')
    ax.axis('off')
    ax.set_title(f"{idx}'{ordinal_indicator.get(idx, 'th')} eigenvector")
fig.tight_layout()
fig.savefig(f'results/{IMAGE_NAME}_eig_vecs.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


# plots segmentations sparse
# fig, axs = plt.subplots(2, 3, figsize=(10, 15))
# metrices = [w_ij, w_i, w_x]
# for row, (similarity_matrix, _axs) in enumerate(zip(metrices, axs.T)):
#     labels, _encodings = cluster(similarity_matrix, k, return_encodings=True)
#     segmented_img = labels.reshape(before_shape)
#     colored_segmentations = colors[segmented_img]
#     colored_image = colored_segmentations * img_gray[:, :, None]

#     _axs[0].imshow(colored_image)
#     _axs[1].imshow(colored_segmentations)

#     if row == 0:
#         _axs[0].set_ylabel('Colored by cluster')
#         _axs[1].set_ylabel('Segmentations')

#     _axs[0].set_xlabel(('Joint', 'Intensity', 'Spatial')[row])
# for ax in axs.flatten():
#     # ax.axis('off')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.xaxis.set_label_position('top')
# fig.tight_layout()
# fig.savefig(f'results/{IMAGE_NAME}_segmented_sparse.pdf', bbox_inches='tight', pad_inches=0)

# show all images
num_imgs = len(params)
fig, axs = plt.subplots(1, num_imgs, figsize=(5*num_imgs, 5))
for ax, img_name in zip(axs.flatten(), params.keys()):
    img_raw = cv2.imread(f'spectral_data/spectral_data/{img_name}.png')
    img_raw = cv2.resize(img_raw, (img_raw.shape[1]//img_downscale, img_raw.shape[0]//img_downscale))

    img_col = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB).astype(float) / 255
    ax.imshow(img_col)
    ax.axis('off')
    ax.set_title(img_name)
fig.tight_layout()
fig.savefig(f'results/all_images.pdf', bbox_inches='tight', pad_inches=0)