import torch
import numpy as np
from scipy.ndimage import uniform_filter

def rgb2gray(rgb):
  """Convert a RGB image to grayscale image
  """
  return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

def get_hog_feature(cuda_imgs, cell_size=16):
  """Compute Histogram of Gradient (HOG) feature for an image

    Modified from skimage.feature.hog
        http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

    Reference:
        Histograms of Oriented Gradients for Human Detection
        Navneet Dalal and Bill Triggs, CVPR 2005

    Parameters:
      im : an input grayscale or rgb image
      cell_size: the cell size to extract features (default 16)

    Returns:
      feat: Histogram of Gradient (HOG) feature, 
        shape = (imx / sx) * (imy / sy) * 9 (default 1764 x 1)
  """
  images = cuda_imgs.cpu()
  hog = torch.empty((images.shape[0], 1764))

  for idx in range(images.shape[0]):
    im = images[idx]
    # convert rgb to grayscale if needed
    image = rgb2gray(im.permute(1,2,0))

    sx, sy = image.shape  # image size
    orientations = 9  # number of gradient bins
    cx, cy = (cell_size, cell_size) # pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[
            round(cx / 2) :: cx, round(cy / 2) :: cy
        ].T
    hog[idx] = torch.from_numpy(orientation_histogram.ravel())

  return hog