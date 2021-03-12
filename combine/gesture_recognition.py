import os
import numpy as np
import cv2
import random
from inference import DetectorYolov5, Classifier
from .base import CombineBase


class GestureRecognition(CombineBase):
    def __init__(self, det_model_file, cls_model_file, det_input_size, cls_input_size, det_conf_thr=0.5, det_iou_thr=0.45):

        # load detection model 
        self._detector = DetectorYolov5(
            det_model_file, input_size=det_input_size, conf_thres=det_conf_thr, iou_thres=det_iou_thr)
        
        # load classifier model
        self._classifier = Classifier(cls_model_file, input_size=cls_input_size)

        # idx -> classes
        self.idx2classes = dict(enumerate(['0', 'close', 'open']))
    
    def _single_frame(self, img):
        out = self._detector.forward(img)
        if out.shape[0] < 1:
            return None
        categorys = []
        objs = []
        for _, dr in enumerate(out):
            hand_image = img[dr[1]: dr[3], dr[0]: dr[2], :]
            if hand_image.shape[0] < 10 or hand_image.shape[1] < 10:
                continue
            categorys.append(self._classifier.forward(hand_image))
            objs.append(dr)
        return objs, categorys

    def _visual(self, frame, objs, categorys):
        if objs:
            for i, dr in enumerate(objs):
                if categorys[i] == 0:
                    continue
                elif categorys[i] == 1:
                    # elif dr[-1] == 1 or dr[-1] == 2:
                    cv2.rectangle(frame, (dr[0], dr[1]), (dr[2], dr[3]), (0, 0, 255), 2, 1)
                    cv2.putText(frame, self.idx2classes[categorys[i]], (dr[0], dr[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                else:
                    cv2.rectangle(frame, (dr[0], dr[1]), (dr[2], dr[3]), (0, 255, 0), 2, 1)
                    cv2.putText(frame, self.idx2classes[categorys[i]], (dr[0], dr[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        return frame

    def video_demo(self, video_file, out_root=None, is_show=False):
        if out_root and not os.path.exists(out_root):
            os.makedirs(out_root)
        frame_iter = self._video(video_file)
        fps, h, w = next(frame_iter)
        # self._video 之后生成 self.ofps, self.ow, self.oh
        video_writer = cv2.VideoWriter(os.path.join(out_root, os.path.basename(video_file)), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        while True:
            try:
                frame = next(frame_iter)
                # 获取检测和关键点推理的结果
                out = self._single_frame(frame[:, :, ::-1])
                if out is not None:
                    objs, categorys = out
                    if len(objs) < 1:
                        continue
                    # 可视化结果
                    frame = self._visual(frame, objs=objs, categorys=categorys)
                video_writer.write(frame)
                if is_show:
                    cv2.imshow("demo", frame)
                    cv2.waitkey(5)
            except StopIteration as e:
                print('Done!')
                break
        video_writer.release()

    def image_demo(self, path, out_root=None, is_show=False):
        if out_root and not os.path.exists(out_root):
            os.makedirs(out_root)
        out = self._single_image(path)
        if out is not None:
            (objs, categorys), frame = out
            if len(objs) < 1:
                return None
            # 可视化结果
            frame = self._visual(frame, objs=objs, categorys=categorys)
        else:
            return None
        if is_show:
            cv2.imshow("demo", frame)
            cv2.waitkey(5000)

        if out_root:
            cv2.imwrite(os.path.join(out_root, os.path.basename(path)), frame)

