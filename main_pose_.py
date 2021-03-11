from combine import HumanPose
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

    opt = parser.parse_args()
    print(opt)

    out_root = opt.out_root
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    # init model
    if not opt.type in ['darkpose', 'baseline']:
        raise Exception("Unsupported type {}".format(opt.type))

    pose = HumanPose(opt.det_model_path, opt.pose_model_path, det_input_size=opt.det_input_size, pose_input_size=opt.pose_input_size,
                     pose_type=opt.type, det_conf_thr=opt.det_conf_thr, det_iou_thr=opt.det_iou_thr, pose_thr=opt.show_thr)

    if opt.video_path:
        pose.video_demo(opt.video_path, opt.out_root)

    if opt.im_path:
        pose.image_demo(opt.im_path, opt.out_root, is_show=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model_path", type=str,
                        default="weights/yolov5s.onnx", help="onnx model path")
    parser.add_argument("--pose_model_path", type=str, default="weights/hrnet_w32_dark_128x96_trim.onnx", help="onnx model path")
    # parser.add_argument("--pose_model_path", type=str, default="weights/hrnet_w32_dark_256x192.onnx", help="onnx model path")
    # parser.add_argument("--pose_model_path", type=str, default="weights/lpn_50_256x192.onnx", help="onnx model path")
    parser.add_argument("--det_input_size", type=tuple, default=640, help="input size")
    parser.add_argument("--pose_input_size", type=tuple, default=(128, 96), help="input size")
    # parser.add_argument("--pose_input_size", type=tuple, default=(256, 192), help="input size")
    parser.add_argument("--show_thr", type=float, default=0.2,
                        help="Threshold that needs to be displayed")
    parser.add_argument("--det_conf_thr", type=float, default=0.2,
                        help="Det threshold that needs to be displayed")
    parser.add_argument("--det_iou_thr", type=float,
                        default=0.45, help="Det threshold that iou")
    parser.add_argument("--type", type=str, default="darkpose",
                        help="Currently supports ['darkpose', 'baseline'], they difference is post-processing")
    parser.add_argument("--video_path", type=str,
                        default="", help="video path")
    parser.add_argument("--im_path", type=str,
                        default="data/person/002.jpg", help="single image path")
    parser.add_argument("--out_root", type=str,
                        default="data/main_result/pose", help="result output folder")
    main()
