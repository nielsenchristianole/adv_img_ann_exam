import cv2
import numpy as np
import matplotlib.pyplot as plt


IMG_FRAC = 10
radius = 30
sigma_i = 10
sigma_x = 20

# load image
img = cv2.imread('spectral_data/spectral_data/plane.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
img = cv2.resize(img, (img.shape[1]//IMG_FRAC, img.shape[0]//IMG_FRAC))

# get params
before_shape = img.shape
mat_lenth = np.prod(img.shape)
img = img.reshape(-1)

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
int_diffs = np.abs(img[row_inx] - img[col_idx])

# calculate the intensity, spatial and joint weights
w_i = (np.exp(-int_diffs / sigma_i)).reshape(mat_lenth, mat_lenth)
w_x = (np.exp(-dist / sigma_x) * mask).reshape(mat_lenth, mat_lenth)

w_ij = np.exp((-int_diffs / sigma_i) + (-dist / sigma_x)) * mask
w_ij = w_ij.reshape(mat_lenth, mat_lenth)

# visualize the weights matrices
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(w_i, cmap='gray')
axs[1].imshow(w_x, cmap='gray')
axs[2].imshow(w_ij, cmap='gray')
axs[0].set_title('Intensity')
axs[1].set_title('Spatial')
axs[2].set_title('Joint')

plt.show()

eig_vals, eig_vecs = np.linalg.eigh(w_ij)

