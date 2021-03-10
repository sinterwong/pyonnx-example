from inference import DetectorYolov5
import glob
import cv2
from tqdm import tqdm
from imutils import paths
import numpy as np
import os


def main():
    # weights = "weights/yolov5s.onnx"
    # detector = DetectorYolov5(weights, input_size=640, conf_thres=0.2, iou_thres=0.45)
    weights = "weights/hand-yolov5-640.onnx"
    detector = DetectorYolov5(weights, input_size=640, conf_thres=0.35, iou_thres=0.45)

    im_path = "data/det/zidane.jpg"
    img = cv2.imread(im_path)[:, :, ::-1]
    show_img = img.copy()

    out = detector.forward(img)

    for i, dr in enumerate(out):
        cv2.rectangle(show_img, (dr[0], dr[1]), (dr[2], dr[3]), 255, 2, 1)
        # crop_img = img[o[1]: o[3], o[0]: o[2], :]
        # cv2.imwrite("%03d.jpg" % i, crop_img[:, :, ::-1])
    cv2.imwrite("detect_out.jpg", show_img[:, :, ::-1])
    
    # 测速
    # imgs = [img] * 100
    # e1 = cv2.getTickCount()
    # for im in imgs:
    #     out = detector.forward(img)
    # e2 = cv2.getTickCount()
    # time = (e2 - e1) / cv2.getTickFrequency()
    # # 关闭视频文件
    # print("总耗时：{}s".format(time))
    # print("单帧耗时：{}s".format(time / 100.))


if __name__ == "__main__":
    main()
