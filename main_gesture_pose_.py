from combine import GestureRecognitionPose
from tools.visualize import visualize
import glob
import cv2
from tqdm import tqdm
from imutils import paths
import numpy as np
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "8"


def main():
    """
    Gesture estimation demo
    """

    opt = parser.parse_args()
    print(opt)

    out_root = opt.out_root
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    # init model
    gesture = GestureRecognitionPose(opt.det_model_path, opt.pose_model_path, det_input_size=opt.det_input_size,
                                     pose_input_size=opt.pose_input_size, angle_thr=opt.angle_thr, 
                                     det_conf_thr=opt.det_conf_thr, det_iou_thr=opt.det_iou_thr)
    if opt.video_path:
        gesture.video_demo(opt.video_path, opt.out_root)

    if opt.im_path:
        gesture.image_demo(opt.im_path, opt.out_root, is_show=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model_path", type=str,
                        default="weights/handv2-yolov5-320.onnx", help="det onnx model path")
    parser.add_argument("--pose_model_path", type=str,
                        default="weights/resnet_50-size-256-wingloss102-0.onnx", help="cls onnx model path")
    parser.add_argument("--det_input_size", type=tuple,
                        default=320, help="det input size")
    parser.add_argument("--pose_input_size", type=tuple,
                        default=(256, 256), help="cls input size")
    parser.add_argument("--det_conf_thr", type=float, default=0.3,
                        help="Det threshold that needs to be displayed")
    parser.add_argument("--det_iou_thr", type=float,
                        default=0.25, help="Det threshold that iou")
    parser.add_argument("--angle_thr", type=int,
                        default=90, help="vector angle threshold")
    parser.add_argument("--type", type=str, default="regression",
                        help="Currently supports ['regression']")
    parser.add_argument("--video_path", type=str,
                        default="data/video/gesture_demo2.mp4", help="video path")
    parser.add_argument("--im_path", type=str,
                        default="data/person/008.jpg", help="single image path")
    parser.add_argument("--out_root", type=str,
                        default="data/main_result/gesture2", help="result output folder")
    main()
