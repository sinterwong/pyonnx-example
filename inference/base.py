import time
from pathlib import Path
import os
import cv2
import numpy as np
from numpy import random
import onnx
import onnxruntime
import abc


class ONNXBase(metaclass=abc.ABCMeta):
    def __init__(self, weights, input_size):
        assert os.path.exists(weights), "model file is not found!"
        self.input_size = input_size
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model(weights)

    def _load_model(self, weights):
        self.session = onnxruntime.InferenceSession(weights)
        self.input_name = self._get_input_name(self.session)
        self.output_name = self._get_output_name(self.session)

    def _get_output_name(self, session):
        """
        output_name = session.get_outputs()[0].name
        :param session:
        :return:
        """
        output_name = []
        for node in session.get_outputs():
            output_name.append(node.name)
        return output_name

    def _get_input_name(self, session):
        """
        input_name = session.get_inputs()[0].name
        :param session:
        :return:
        """
        input_name = []
        for node in session.get_inputs():
            input_name.append(node.name)
        return input_name

    def _get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def preprocessing(self, image, aspect_ratio=True, is_norm=True, sub_mean=None, div_std=None):
        rh, rw = None, None
        if aspect_ratio:
            h, w, _ = image.shape
            data = np.zeros(
                [self.input_size, self.input_size, 3], dtype=np.float32)
            ration = float(self.input_size) / max(h, w)
            rh, rw = round(h * ration), round(w * ration)
            data[:rh, :rw, :] = cv2.resize(image, (rw, rh)).astype(np.float32)
        else:
            data = cv2.resize(image, (self.input_size, self.input_size)).astype(np.float32)
            
        if is_norm:
            data /=  255.0

        data = data.transpose([2, 0, 1])
        
        if sub_mean and div_std:
            assert isinstance(sub_mean, int) or len(
                sub_mean) == 3, "vaild args sub_mean"
            for i, (m, s) in enumerate(zip(sub_mean, div_std)):
                data[i] -= m
                data[i] /= s

        return np.expand_dims(data, 0), rw, rh

    @abc.abstractmethod
    def forward(self, image):
        pass
