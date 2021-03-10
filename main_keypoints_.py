from inference import KeypointsDarkPose, KeypointsLPN
from tools.visualize import visualize
import glob
import cv2
from tqdm import tqdm
from imutils import paths
import numpy as np
import os


def main():

    out_root = "data/main_result"
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    weights = "weights/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922_trim.onnx"
    detector = KeypointsDarkPose(weights)

    # weights = "weights/pose_coco/lpn_50_256x192.onnx"
    # detector = KeypointsLPN(weights)

    im_path = "data/person/004.jpg"
    img = cv2.imread(im_path)[:, :, ::-1]
    show_img = img[:, :, ::-1].copy()

    points = detector.forward(img)

    # imgs = [img] * 100
    # e1 = cv2.getTickCount()
    # for im in imgs:
    #     points = detector.forward(im)

    # e2 = cv2.getTickCount()
    # time = (e2 - e1) / cv2.getTickFrequency()
    # # 关闭视频文件
    # print("总耗时：{}s".format(time))
    # print("单帧耗时：{}s".format(time / 100.))

    for i, p in enumerate(points):
        if p[2] < 0.35:
            continue
        cv2.circle(show_img, tuple(p[:2].astype(np.int32)), 1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out_root, 'keypoints_out.jpg'), show_img)


if __name__ == "__main__":
    main()
