import numpy as np
import cv2
from numpy.fft import fft2
from numpy.fft import ifft2
from tracker.utils import gaussian_shaped_labels
from tracker.utils import get_subwindow
from tracker.utils import get_feature
from tracker.utils import gaussian_correlation
from tracker.utils import polynomial_correlation
from tracker.utils import format_box


class KCFtracker(object):

    def __init__(self, img, start_pos, target_size, padding=2.5, lamb=0.0001, output_sigma_factor=0.1,
                 interp_factor=0.075, cell_size=1, feature='gray', resize=False,
                 kernel={'kernel_type': 'gaussian', 'sigma': 0.2}, showvideo=True):
        self.original_img = img
        self.img = img
        self.padding = padding
        self.lamb = lamb
        self.output_sigma_factor = output_sigma_factor
        self.interp_factor = interp_factor
        self.cell_size = cell_size
        self.feature = feature
        self.showvideo = showvideo
        self.original_pos = start_pos
        self.original_target_size = target_size
        # the box's CENTER point coordinate,it is format as [y, x]
        self.pos = start_pos
        # the box's size , it is format as [h, w]
        self.target_size = target_size
        self.base_size = target_size

        self.kernel_type = kernel['kernel_type']
        if self.kernel_type == 'gaussian':
            self.kernel_sigma = kernel['sigma']
        elif self.kernel_type == 'polynomial':
            self.kernel_poly_a = kernel['poly_a']
            self.kernel_poly_b = kernel['poly_b']

        self.resize = resize
        if np.sqrt(np.prod(self.target_size)) >= 150:
            print("resize image")
            self.resize = True

        if self.resize:
            print("image is resized")
            self.pos = tuple([int(ele/2) for ele in self.pos])
            self.target_size = tuple([int(ele/2) for ele in self.target_size])
            self.img_size = (int(img.shape[0]/2), int(img.shape[1]/2))
            img = cv2.resize(img, self.img_size[::-1])

        # in opencv the img's size get from shape
        # the shape is format as (h,w,c), c is the chanel of image
        self.img_size = (img.shape[0], img.shape[1])
        self.window_size = (
            int(self.target_size[0]*self.padding), int(self.target_size[1]*self.padding))

        if self.feature == 'gray':
            # translate image from rgb to gray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 一些不会变的变量，如余弦窗，期望分布
        output_sigma = np.sqrt(np.prod(self.target_size)) * \
            self.output_sigma_factor / self.cell_size
        self.y = gaussian_shaped_labels(output_sigma, (int(
            self.window_size[0]/self.cell_size), int(self.window_size[1]/self.cell_size)))
        self.yf = fft2(self.y)
        self.cos_window = np.outer(np.hanning(
            self.yf.shape[0]), np.hanning(self.yf.shape[1]))

        # 初始化模型
        patch = get_subwindow(img, self.pos, self.window_size)
        xf = fft2(get_feature(patch, self.feature,
                              self.cell_size, self.cos_window))
        self.model_xf = xf
        self.model_alphaf = self.__train(xf)
        self.__show_image()

    def dectect(self, img):
        # 这个其实和下面update是一样的，不知道为什么源代码会分成两部分写
        self.original_img = img
        self.img = img
        if self.resize:
            self.img = cv2.resize(self.img, self.img_size[::-1])

        if self.feature == 'gray':
            # translate image from rgb to gray
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        patch = get_subwindow(self.img, self.pos, self.window_size)
        zf = fft2(get_feature(patch, self.feature,
                              self.cell_size, self.cos_window))

        if self.kernel_type == "gaussian":
            kzf = gaussian_correlation(zf, self.model_xf, self.kernel_sigma)
        elif self.kernel_type == "polynomial":
            kzf = polynomial_correlation(
                zf, self.model_xf, self.kernel_poly_a, self.kernel_poly_b)
            pass

        response = np.real(ifft2(self.model_alphaf * kzf))
        cv2.imshow("response", response)
        cv2.waitKey(10)

        [vert_delta, horiz_delta] = np.unravel_index(
            response.argmax(), response.shape)
        # print("[vert_delta, horiz_delta] = " + str([vert_delta, horiz_delta]))
        if vert_delta > zf.shape[0]/2:
            vert_delta = vert_delta - zf.shape[0]
        if horiz_delta > zf.shape[1]/2:
            horiz_delta = horiz_delta - zf.shape[1]
        # print("after handled [vert_delta, horiz_delta] = " + str([vert_delta, horiz_delta]))

        self.pos = int(self.pos[0] + self.cell_size * (vert_delta)
                       ), int(self.pos[1] + self.cell_size * (horiz_delta))
        self.__show_image()
        self.__update()
        return self.original_pos, self.original_target_size

    def __train(self, xf):
        if self.kernel_type == "gaussian":
            kf = gaussian_correlation(xf, xf, self.kernel_sigma)
        elif self.kernel_type == "polynomial":
            kf = polynomial_correlation(
                xf, xf, self.kernel_poly_a, self.kernel_poly_b)
        alphaf = self.yf / (kf + self.lamb)
        return alphaf

    def __update(self):
        patch = get_subwindow(self.img, self.pos, self.window_size)
        xf = fft2(get_feature(patch, self.feature,
                              self.cell_size, self.cos_window))
        alphaf = self.__train(xf)
        self.model_xf = (1-self.interp_factor) * \
            self.model_xf + self.interp_factor * xf
        self.model_alphaf = (1-self.interp_factor) * \
            self.model_alphaf + self.interp_factor * alphaf

    def __show_image(self, delay=10):
        if self.resize:
            self.original_pos = (self.pos[0]*2, self.pos[1]*2)
            self.original_target_size = (
                self.target_size[0]*2, self.target_size[1]*2)
        else:
            self.original_pos = self.pos
            self.original_target_size = self.target_size

        if self.showvideo:
            x, y, w, h = format_box(
                self.original_pos, self.original_target_size)
            cv2.rectangle(self.original_img, (x, y),
                          (x+w, y+h), (0, 255, 0), 1)
            cv2.imshow("KCF tracker", self.original_img)
            cv2.waitKey(delay)
