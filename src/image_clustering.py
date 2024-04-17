import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


IMAGE_NAME = 'peppers'


params = dict(
    plane = dict(
        img_downscale = 6,
        radius = 100,
        sigma_i = 0.5,
        sigma_x = 145,
        k = 2,
        mode='gray'
    ),
#     bag = dict(
#         img_downscale = 6,
#         radius = 100,
#         sigma_i = 0.5,
#         sigma_x = 145,
#         k = 2
#     ),
    onion = dict(
        img_downscale = 6,
        radius = 100,
        sigma_i = 0.5,
        sigma_x = 145,
        k = 5
    ),
    peppers = dict(
        img_downscale = 6,
        radius = 100,
        sigma_i = 0.5,
        sigma_x = 145,
        k = 5,
        mode='color'
    )
)


img_downscale = params[IMAGE_NAME]['img_downscale']
radius = params[IMAGE_NAME]['radius']
sigma_i = params[IMAGE_NAME]['sigma_i']
sigma_x = params[IMAGE_NAME]['sigma_x']
k = params[IMAGE_NAME]['k']
mode = params[IMAGE_NAME]['mode']


def cluster(similarity_matrix: np.ndarray, k: int) -> np.ndarray:

    degrees = np.count_nonzero(similarity_matrix, axis=1)
    degree_matrix = np.diag(degrees)
    laplacian = degree_matrix - similarity_matrix

    sqrt_degree_matrix = np.diag(1 / np.sqrt(degrees))
    inv_degree_matrix = np.diag(1 / degrees)

    laplacian_sym = sqrt_degree_matrix @ laplacian @ sqrt_degree_matrix
    laplacian_rw = inv_degree_matrix @ laplacian

    evalues, evectors = np.linalg.eigh(laplacian_sym)

    rw_evectors = sqrt_degree_matrix @ evectors

    U = evectors[:, :k]
    U = U / np.linalg.norm(U, axis=1, keepdims=True)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
    return kmeans.labels_


# load image
img_raw = cv2.imread(f'spectral_data/spectral_data/{IMAGE_NAME}.png')
img_raw = cv2.resize(img_raw, (img_raw.shape[1]//img_downscale, img_raw.shape[0]//img_downscale))

img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY).astype(float) / 255
img_col = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB).astype(float) / 255

img = img_gray if mode == 'gray' else img_col

# get params
before_shape = img_gray.shape
mat_lenth = np.prod(before_shape)
if mode == 'gray':
    img = img.reshape(-1)[..., None]
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
w_i = (np.exp(-int_diffs / sigma_i)).reshape(mat_lenth, mat_lenth)
w_x = (np.exp(-dist / sigma_x) * mask).reshape(mat_lenth, mat_lenth)

w_ij = np.exp((-int_diffs / sigma_i) + (-dist / sigma_x)) * mask
w_ij = w_ij.reshape(mat_lenth, mat_lenth)

# visualize the weights matrices
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# axs[0].imshow(w_i, cmap='gray')
# axs[1].imshow(w_x, cmap='gray')
# axs[2].imshow(w_ij, cmap='gray')
# axs[0].set_title('Intensity')
# axs[1].set_title('Spatial')
# axs[2].set_title('Joint')
# for ax in axs:
#     ax.axis('off')
# fig.tight_layout()
# plt.show()

colors = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
], dtype=float)


fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs = axs if len(axs.shape) == 2 else np.array([axs])

for row, (similarity_matrix, _axs) in enumerate(zip([w_ij, w_i, w_x], axs)):
    labels = cluster(similarity_matrix, k)
    segmented_img = labels.reshape(before_shape)
    colored_segmentations = colors[segmented_img]
    colored_image = colored_segmentations * img_gray[:, :, None]

    if mode == 'gray':
        _axs[0].imshow(img_gray, cmap='gray')
    else:
        _axs[0].imshow(img_col)
    _axs[1].imshow(colored_image)
    _axs[2].imshow(colored_segmentations)

    if row == 0:
        _axs[0].set_xlabel('Original')
        _axs[1].set_xlabel('Colored by cluster')
        _axs[2].set_xlabel('Segmentations')
    
    _axs[0].set_ylabel(('Intensity', 'Spatial', 'Joint')[row])

for ax in axs.flatten():
    # ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_label_position('top')


fig.tight_layout()
fig.savefig(f'results/{IMAGE_NAME}_segmented.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
