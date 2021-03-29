from combine import GestureRecognition
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

    # idx2classes = dict(enumerate(['one', 'five', 'fist', 'ok', 'heartSingle',
    #                               'yearh', 'three', 'four', 'six', 'Iloveyou', 'gun', '-thumbUp', 'nine', 'pink']))
    idx2classes = dict(enumerate(['0', 'close', 'open']))
    # init model
    gesture = GestureRecognition(opt.det_model_path, opt.cls_model_path, det_input_size=opt.det_input_size, cls_input_size=opt.cls_input_size,
                                 idx2classes=idx2classes, det_conf_thr=opt.det_conf_thr, det_iou_thr=opt.det_iou_thr, is_tracker=True)

    if opt.video_path:
        gesture.video_demo(opt.video_path, opt.out_root)

    if opt.im_path:
        gesture.image_demo(opt.im_path, opt.out_root, is_show=False)

    if opt.action_video_path:
        # 观测区域，启动动作只需要针对这块区域的手部就可以
        field_view = [50, 300, 300, 620]  # x1, y1, x2, y2
        gesture.action_demo(opt.action_video_path, field_view=field_view, out_root=opt.out_root, is_show=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model_path", type=str,
                        default="weights/handv2-yolov5-320.onnx", help="det onnx model path")
    parser.add_argument("--cls_model_path", type=str, default="weights/hand-recognition_0.992_3.onnx", help="cls onnx model path")
    # parser.add_argument("--cls_model_path", type=str, default="weights/hand-recognition_resnet10_85.714.onnx", help="cls onnx model path")
    parser.add_argument("--det_input_size", type=tuple,
                        default=320, help="det input size")
    parser.add_argument("--cls_input_size", type=tuple,
                        default=64, help="cls input size")
    parser.add_argument("--det_conf_thr", type=float, default=0.3,
                        help="Det threshold that needs to be displayed")
    parser.add_argument("--det_iou_thr", type=float,
                        default=0.25, help="Det threshold that iou")
    parser.add_argument("--video_path", type=str,
                        default="", help="video path")
    parser.add_argument("--im_path", type=str,
                        default="", help="single image path")
    parser.add_argument("--action_video_path", type=str,
                        default="data/video/action/palm_move.mp4", help="single image path")
    parser.add_argument("--out_root", type=str,
                        default="data/main_result/gesture", help="result output folder")
    main()
