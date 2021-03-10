import numpy as np
import cv2
from numpy.fft import fft2
from numpy.fft import ifft2
# from fhog import getFeatureMaps,normalizeAndTruncate,PCAFeatureMaps


def gaussian_shaped_labels(sigma, window_size):
    # window_size = (h,w) w is the windows's width,h is the height
    (h, w) = window_size
    r = np.arange(1, w+1) - int(w/2)
    s = np.arange(1, h+1) - int(h/2)
    [rs, cs] = np.meshgrid(r, s)

    labels = np.exp(-0.5/(sigma**2) * (rs**2+cs**2))
    # 原论文附录A1提到
    labels = np.roll(labels, -int(w/2)+1, axis=1)
    labels = np.roll(labels, -int(h/2)+1, axis=0)
    return labels


def get_subwindow(img, pos, window_size, resize=None):
    ih, iw = img.shape[0], img.shape[1]
    h, w = window_size
    y, x = pos[0], pos[1]

    xl = x - int(w/2)
    yl = y - int(h/2)
    xr = xl + w
    yr = yl + h

    cop_xl = max(xl, 0)
    cop_yl = max(yl, 0)
    cop_xr = min(xr, iw)
    cop_yr = min(yr, ih)

    left = -xl if xl < 0 else 0
    top = -yl if yl < 0 else 0
    right = xr-(iw) if xr > iw else 0
    bottom = yr-(ih) if yr > ih else 0

    patch = img[cop_yl:cop_yr, cop_xl:cop_xr]
    if len(img.shape) == 3:
        patch = np.pad(patch, ((top, bottom), (left, right), (0, 0)), 'edge')
    else:
        patch = np.pad(patch, ((top, bottom), (left, right)), 'edge')

    return patch


# def get_hog_feature(img,cell_size):
#     mapp = {'sizeX':0, 'sizeY':0, 'numFeatures':0, 'map':0}
#     mapp = getFeatureMaps(img, cell_size, mapp)
#     mapp = normalizeAndTruncate(mapp, 0.2)
#     mapp = PCAFeatureMaps(mapp)
#     size_patch = map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']])
#     size_patch = list(size_patch)
#     FeaturesMap = mapp['map'].reshape((size_patch[0],size_patch[1], size_patch[2])).T
#     return FeaturesMap


def get_feature(patch, feature, cell_size, cos_window=None):

    if feature == 'gray':
        patch = patch.astype('float64') / 255
        patch = patch - np.mean(patch)

    elif feature == 'rgb':
        patch = patch.astype('float64') / 255
        for i in range(patch.shape[2]):
            patch[:, :, i] -= np.mean(patch[:, :, i])
    elif feature == 'hog':
        # patch = get_hog_feature(patch,cell_size)
        pass

    if len(patch.shape) == 2:
        patch = patch * cos_window
    else:
        patch = patch * cos_window[:, :, np.newaxis]

    return patch


def gaussian_correlation(xf, yf, sigma):

    N = xf.shape[0] * xf.shape[1]

    x_ = np.asmatrix(xf.flatten())
    y_ = np.asmatrix(yf.flatten())
    xx = np.dot(x_, x_.getH()) / N
    yy = np.dot(y_, y_.getH()) / N

    xyf = xf * np.conj(yf)
    if len(xyf.shape) == 3:
        xy = np.sum(np.real(ifft2(xyf)), axis=2)
    else:
        xy = np.real(ifft2(xyf))
    kf = fft2(np.exp((-1.0 / sigma**2) *
                     np.maximum((xx+yy-2*xy) / float(np.size(xf)), 0)))
    return kf


def polynomial_correlation(xf, yf, kernel_poly_a, kernel_poly_b):
    kf = []
    return kf


def format_box(pos, target_size):
    h = target_size[0]
    w = target_size[1]
    y = int(pos[0] - h/2)
    x = int(pos[1] - w/2)
    return x, y, w, h
