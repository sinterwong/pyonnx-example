from pathlib import Path
import os
import cv2
import numpy as np
from numpy import random
from .detector import Detector


class DetectorYolov5(Detector):
    def __init__(self, weights, input_size=640, conf_thres=0.2, iou_thres=0.45, classes=0, agnostic_nms=False):
        assert os.path.exists(weights), "model file is not found!"
        self.input_size = input_size
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model(weights)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms

    def forward(self, image):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        h, w, _ = image.shape

        # TODO 以下注释为测速
        # pre_start = cv2.getTickCount()
        data, rw, rh = self.preprocessing(image)
        # pre_end = cv2.getTickCount()
        # print("前处理耗时：{}s".format((pre_end - pre_start) / cv2.getTickFrequency()))

        # forward_start = cv2.getTickCount()
        input_feed = self._get_input_feed(self.input_name, data)
        pred = self.session.run(
            self.output_name, input_feed=input_feed)[0]
        # forward_end = cv2.getTickCount()
        # print("前传耗时：{}s".format((forward_end - forward_start) / cv2.getTickFrequency()))

        # post_start = cv2.getTickCount()
        out = self.non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms)[0]
        out[:, 0] = out[:, 0] / rw * w
        out[:, 1] = out[:, 1] / rh * h
        out[:, 2] = out[:, 2] / rw * w
        out[:, 3] = out[:, 3] / rh * h
        # post_end = cv2.getTickCount()
        # print("后处理耗时：{}s".format((post_end - post_start) / cv2.getTickFrequency()))
        out[out[:, 0] < 0] = 0
        out[out[:, 1] < 0] = 0
        out[out[:, 2] < 0] = 0
        out[out[:, 3] < 0] = 0
        return out.astype(np.int32)


