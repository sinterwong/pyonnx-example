from pathlib import Path
import os
import cv2
import numpy as np
from ..keypoints import Keypoints
from tools.pose_transforms import xywh2cs


class KeypointsReg(Keypoints):
    def __init__(self, weights, input_size=(256, 256), conf_thres=0.2):
        assert os.path.exists(weights), "model file is not found!"
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model(weights)

    def preprocessing(self, image):
        data = cv2.resize(image, (self.input_size[1], self.input_size[0])).astype(np.float32)
        data -= 128.
        data /= 256.
        data = data.transpose([2, 0, 1])
        
        return np.expand_dims(data, 0)

    def forward(self, image):
        '''
        :param image: (RGB)(H, W, C)
        :return:
        '''
        h, w, _ = image.shape

        data = self.preprocessing(image)

        # forward_start = cv2.getTickCount()
        input_feed = self._get_input_feed(self.input_name, data)
        outputs = self.session.run(self.output_name, input_feed=input_feed)[0]
        # forward_end = cv2.getTickCount()
        # print("推理耗时：{}s".format((forward_end - forward_start) / cv2.getTickFrequency()))

        # post_start = cv2.getTickCount()
        preds = self._get_final_preds_reg(outputs, w, h)
        # post_end = cv2.getTickCount()
        # print("后处理耗时：{}s".format((post_end - post_start) / cv2.getTickFrequency()))

        return preds
