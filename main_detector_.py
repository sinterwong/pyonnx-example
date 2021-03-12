from inference import DetectorYolov5
import glob
import cv2
from tqdm import tqdm
from imutils import paths
import numpy as np
import os
import argparse


def main():
    """
    detection Demo
    """
    opt = parser.parse_args()
    print(opt)

    out_root = opt.out_root
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    # init model
    if not opt.type in ['yolo']:
        raise Exception("Unsupported type {}".format(opt.type)) 
    
    if opt.type == 'yolo':
        detector = DetectorYolov5(opt.model_path, input_size=opt.input_size, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)

    img = cv2.imread(opt.im_path)[:, :, ::-1]
    show_img = img[:, :, ::-1].copy()

    # inference
    out = detector.forward(img)

    for i, dr in enumerate(out):
        cv2.rectangle(show_img, (dr[0], dr[1]), (dr[2], dr[3]), 255, 2, 1)
        if opt.crop_obj:
            crop_img = img[dr[1]: dr[3], dr[0]: dr[2], :]
            if not os.path.exists(os.path.join(out_root, "crop_res", coco_idx2classes[dr[5]])):
                os.makedirs(os.path.join(out_root, "crop_res", coco_idx2classes[dr[5]]))
            p = os.path.join(out_root, "crop_res", coco_idx2classes[dr[5]], "%03d.jpg" % i)
            cv2.imwrite(p, crop_img[:, :, ::-1])
    cv2.imwrite(os.path.join(out_root, os.path.basename(opt.im_path)), show_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="weights/yolov5s.onnx", help="onnx model path")
    parser.add_argument("--input_size", type=int, default=640, help="input size")
    parser.add_argument("--conf_thres", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="IOU threshold")
    # parser.add_argument("--model_path", type=str, default="weights/pose_coco/lpn_50_256x192.onnx", help="onnx model path")
    parser.add_argument("--type", type=str, default="yolo", help="Currently supports ['yolo'], they difference is post-processing")
    parser.add_argument("--crop_obj", action='store_true', help="Save the detected object")
    parser.add_argument("--im_path", type=str, default="data/det/zidane.jpg", help="single image path")
    parser.add_argument("--out_root", type=str, default="data/main_result/detection", help="result output folder")

    coco_idx2classes = dict(enumerate(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                                    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
                                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
                                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
                                    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
                                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                                    'hair drier', 'toothbrush']))

    hand_idx2classes = dict(enumerate(['hand']))

    main()
