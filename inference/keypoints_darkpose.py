from pathlib import Path
import os
import cv2
import numpy as np
from .keypoints import Keypoints
from tools.pose_transforms import xywh2cs


class KeypointsDarkPose(Keypoints):
    def __init__(self, weights, input_size=(256, 192), conf_thres=0.2):
        assert os.path.exists(weights), "model file is not found!"
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model(weights)

    def forward(self, image):
        '''
        :param image: (RGB)(H, W, C)
        :return:
        '''
        h, w, _ = image.shape

        c, s = xywh2cs(0, 0, w, h)

        data = self.preprocessing(image, c, s, sub_mean=[0.485, 0.456, 0.406], div_std=[0.229, 0.224, 0.225])

        forward_start = cv2.getTickCount()
        input_feed = self._get_input_feed(self.input_name, data)
        outputs = self.session.run(self.output_name, input_feed=input_feed)[0]
        forward_end = cv2.getTickCount()
        print("推理耗时：{}s".format(
            (forward_end - forward_start) / cv2.getTickFrequency()))

        post_start = cv2.getTickCount()
        preds, maxvals = self._get_final_preds_darkpose(outputs, [c], [s])
        # preds, maxvals = self._get_final_preds(outputs, [c], [s])
        # preds[:, maxvals.squeeze() < self.conf_thres] = -1
        preds = np.concatenate([preds, maxvals], axis=2).squeeze()
        # 原始尺度
        # rh = self.input_size[0] / h
        # rw = self.input_size[1] / w
        # preds[..., 1] = preds[..., 1] * \
        #     (self.input_size[0] / outputs.shape[2]) / rh
        # preds[..., 0] = preds[..., 0] * \
        #     (self.input_size[1] / outputs.shape[3]) / rw
        post_end = cv2.getTickCount()
        print("后处理耗时：{}s".format((post_end - post_start) / cv2.getTickFrequency()))

        return preds


if __name__ == "__main__":
    weights = "../weights/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.onnx"
    detector = KeypointsDarkPose(weights)
    im_path = "../data/person/000.jpg"
    img = cv2.imread(im_path)[:, :, ::-1]
    show_img = img[:, :, ::-1].copy()

    imgs = [img] * 100
    e1 = cv2.getTickCount()
    for im in imgs:
        points = detector.forward(im)

    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    # 关闭视频文件
    print("总耗时：{}s".format(time))
    print("单帧耗时：{}s".format(time / 100.))

    for i, p in enumerate(points):
        cv2.circle(show_img, tuple(p[:2].astype(np.int32)), 5, (0, 0, 255), 2)
    cv2.imwrite('out2.jpg', show_img)
