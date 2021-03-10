from inference import Classifier
import glob
import cv2
from tqdm import tqdm
from imutils import paths
import numpy as np
import os
import argparse



def main():
    opt = parser.parse_args()
    print(opt)

    out_root = opt.out_root
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    # init model
    classifier = Classifier(opt.model_path, input_size=opt.input_size)

    # read image
    img = cv2.imread(opt.im_path)[:, :, ::-1]
    show_img = img[:, :, ::-1].copy()

    # inference
    out = classifier.forward(img)

    # save result
    if not os.path.exists(os.path.join(out_root,  hand_idx2classes[out])):
        os.makedirs(os.path.join(out_root, hand_idx2classes[out]))
    p = os.path.join(out_root, hand_idx2classes[out], os.path.basename(opt.im_path))
    cv2.imwrite(p, show_img)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="weights/hand-recognition_0.992_3.onnx", help="onnx model path")
    # parser.add_argument("--model_path", type=str, default="weights/pose_coco/lpn_50_256x192.onnx", help="onnx model path")
    parser.add_argument("--input_size", type=int, default=64, help="input size")
    parser.add_argument("--im_path", type=str, default="data/hand/close.jpg", help="single image path")
    parser.add_argument("--out_root", type=str, default="data/main_result/classifier", help="result output folder")

    hand_idx2classes = dict(enumerate(['0', 'close', 'open']))

    main()
