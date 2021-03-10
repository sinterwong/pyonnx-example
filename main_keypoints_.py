from inference import KeypointsDarkPose, KeypointsLPN
from tools.visualize import visualize
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
    if not opt.type in ['darkpose', 'baseline']:
        raise Exception("Unsupported type {}".format(opt.type)) 
    
    if opt.type == 'darkpose':
        detector = KeypointsDarkPose(opt.model_path)
    elif opt.type == 'baseline':
        detector = KeypointsLPN(opt.weights)

    img = cv2.imread(opt.im_path)[:, :, ::-1]
    show_img = img[:, :, ::-1].copy()

    points = detector.forward(img)

    for i, p in enumerate(points):
        if p[2] < opt.show_thr:
            continue
        cv2.circle(show_img, tuple(p[:2].astype(np.int32)), 1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out_root, os.path.basename(opt.im_path)), show_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="weights/hrnet_w32_dark_256x192.onnx", help="onnx model path")
    parser.add_argument("--show_thr", type=float, default=0.35, help="Threshold that needs to be displayed")
    # parser.add_argument("--model_path", type=str, default="weights/pose_coco/lpn_50_256x192.onnx", help="onnx model path")
    parser.add_argument("--type", type=str, default="darkpose", help="Currently supports ['darkpose', 'baseline'], they difference is post-processing")
    parser.add_argument("--im_path", type=str, default="data/person/004.jpg", help="single image path")
    parser.add_argument("--out_root", type=str, default="data/main_result/keypoints", help="result output folder")

    main()
