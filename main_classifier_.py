from inference import Classifier
import glob
import cv2
from tqdm import tqdm
from imutils import paths
import numpy as np
import os


def main():
    weights = "/home/wangjq/wangxt/workspace/pyonnx-example/weights/hand-recognition_0.994.onnx"
    im_path = "/home/wangjq/wangxt/workspace/pyonnx-example/inference/000.jpg"

    classifier = Classifier(weights, input_size=64)
    img = cv2.imread(im_path)[:, :, ::-1]

    out = classifier.forward(img)

    # 模拟测速
    # imgs = [img] * 100
    # e1 = cv2.getTickCount()
    # for im in imgs:
    #     out = classifier.forward(img)
    # e2 = cv2.getTickCount()
    # time = (e2 - e1) / cv2.getTickFrequency()
    # # 关闭视频文件
    # print("总耗时：{}s".format(time))
    # print("单帧耗时：{}s".format(time / 100.))

if __name__ == "__main__":
    main()
