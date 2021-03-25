import os
import numpy as np
import cv2
import random
from collections import Counter
from inference import DetectorYolov5, Classifier
from tracker import KCFtracker
from .base import CombineBase


class GestureRecognition(CombineBase):
    def __init__(self, det_model_file, cls_model_file, det_input_size, cls_input_size, idx2classes, det_conf_thr=0.5, det_iou_thr=0.45, is_tracker=True):

        # load detection model
        self._detector = DetectorYolov5(
            det_model_file, input_size=det_input_size, conf_thres=det_conf_thr, iou_thres=det_iou_thr)

        # load classifier model
        self._classifier = Classifier(
            cls_model_file, input_size=cls_input_size)

        # idx -> classes
        self.idx2classes = idx2classes

        self.is_tracker = is_tracker

        if self.is_tracker:
            self.tracker = KCFtracker()

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
                if self.idx2classes[categorys[i]] == "0":
                    continue
                cv2.rectangle(frame, (dr[0], dr[1]),
                              (dr[2], dr[3]), (0, 0, 255), 3, 1)
                cv2.putText(frame, self.idx2classes[categorys[i]], (
                    dr[0], dr[1] + dr[3] // 2), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
                # if categorys[i] == 0:
                #     continue
                # elif categorys[i] == 1:
                #     # elif dr[-1] == 1 or dr[-1] == 2:
                #     cv2.rectangle(frame, (dr[0], dr[1]),
                #                   (dr[2], dr[3]), (0, 0, 255), 3, 1)
                #     cv2.putText(frame, self.idx2classes[categorys[i]], (
                #         dr[0], dr[1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
                # else:
                #     cv2.rectangle(frame, (dr[0], dr[1]),
                #                   (dr[2], dr[3]), (0, 255, 0), 3, 1)
                #     cv2.putText(frame, self.idx2classes[categorys[i]], (
                #         dr[0], dr[1]), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)
        return frame

    def video_demo(self, video_file, out_root=None, is_show=False):
        if out_root and not os.path.exists(out_root):
            os.makedirs(out_root)
        frame_iter = self._video(video_file)
        fps, h, w = next(frame_iter)
        # self._video 之后生成 self.ofps, self.ow, self.oh
        video_writer = cv2.VideoWriter(os.path.join(out_root, os.path.basename(
            video_file)), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
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

    def video_tracker_demo(self, video_file, out_root=None, is_show=False):
        if out_root and not os.path.exists(out_root):
            os.makedirs(out_root)
        frame_iter = self._video(video_file)
        fps, h, w = next(frame_iter)
        video_writer = cv2.VideoWriter(os.path.join(out_root, os.path.basename(
            video_file)), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

        count = -1
        objs = None
        categorys = None
        while True:
            try:
                frame = next(frame_iter)
                count += 1
                if self.is_tracker:
                    if (count) % 10 == 0:
                        out = self._single_frame(frame[:, :, ::-1])
                        if out is not None:
                            objs, categorys = out
                            if len(objs) < 1:
                                continue
                            # TODO 仅跟踪一个目标
                            self.tracker.init(frame, objs[0][0:4])
                            frame = self._visual(
                                frame, objs=objs, categorys=categorys)
                    else:
                        if objs is not None:
                            x, y, w, h = self.tracker.update(frame)
                            cv2.rectangle(
                                frame, (x, y), (x + w, y + h), (100, 0, 100), 2, 1)
                            cv2.imwrite('out.jpg', frame)
                        else:
                            out = self._single_frame(frame[:, :, ::-1])
                else:
                    out = self._single_frame(frame[:, :, ::-1])
                    if out is not None:
                        objs, categorys = out
                        if len(objs) < 1:
                            continue
                        # TODO 仅跟踪一个目标
                        self.tracker.init(frame, objs[0][0:4])
                        frame = self._visual(
                            frame, objs=objs, categorys=categorys)

                video_writer.write(frame)
                if is_show:
                    cv2.imshow("demo", frame)
                    cv2.waitkey(5)
            except StopIteration as e:
                print('Done!')
                break
        video_writer.release()

    def _iof(self, box1, box2):
        """ 计算被包含率
            (box1 ∩ box2) / box2
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the area of intersection rectangle
        inter_area = abs(max((x2 - x1, 0)) * max((y2 - y1), 0))
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / box2_area

    def _filter_obj(self, objs, categorys, iof_thr=0.7):
        """ 过滤掉类别为 0 和 不在观测区域内的bbox
        """
        valid_objs = []
        valid_categorys = []

        for i, dr in enumerate(objs):
            if self.idx2classes[categorys[i]] == "0":
                continue
            if self._iof(self.field_view, dr) < iof_thr:
                continue 
            valid_objs.append(dr)
            valid_categorys.append(categorys[i])
        return valid_objs, valid_categorys
    
    def _start_func(self, start_ls, r=0.8):
        """ 是否满足启动条件, 以及是何种动作的预备式
        Args: 
            start_ls(list((obj), (category))): 启动序列
            r(float):  多数的类别必须在此比例之上
        Return:
            action_type: 预备式是何动作
        """
        action_type = None
        objs, categorys = zip(*start_ls)

        # # TODO 模拟测试，用完删除
        # objs = list(objs)
        # categorys = list(categorys)
        # objs.append(objs[-1])
        # objs.append(objs[-1])
        # objs.append(objs[-1])
        # categorys.append(1)
        # categorys.append(3)
        # categorys.append(3)

        num_type = Counter(categorys)
        if len(num_type) == 1:
            action_type = (list(num_type)[0], objs[-1])
        else:
            cs, num_cs = zip(*num_type.items())
            idx = np.argmax(num_cs)
            num_max = num_cs[idx]
            other = sum(num_cs) - num_max
            if (num_max * 1.0 / other) > r:
                 action_type = (cs, objs[-1])

        return action_type



    def action_demo(self, video_file, field_view, out_root=None, is_show=False):
        self.field_view = field_view
        if out_root and not os.path.exists(out_root):
            os.makedirs(out_root)
        frame_iter = self._video(video_file)
        fps, h, w = next(frame_iter)
        # self._video 之后生成 self.ofps, self.ow, self.oh
        video_writer = cv2.VideoWriter(os.path.join(out_root, os.path.basename(
            video_file)), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        
        # 启动序列, 大于多少时启动判罚
        is_start, sn = [], 20
        tracker_state = False
        start_type = None
        # 手部移动的中心点
        real_cp = None
        # 调整量
        up_val = 0
        while True:
            try:
                frame = next(frame_iter)
                if len(is_start) > sn:
                    action_type = self._start_func(is_start)
                    if action_type is not None:
                        # 动作已经启动, 加入跟踪器
                        start_type, obj = action_type
                        # 获取中心点
                        real_cp = [obj[0] + (obj[2] - obj[0]) / 2, obj[1] + (obj[3] - obj[1]) / 2]
                        self.tracker.init(frame, obj[:4])
                        tracker_state = True
                        # 状态判断完成之后回到初始化
                    is_start = []
                if not tracker_state:
                    # 获取检测和关键点推理的结果
                    out = self._single_frame(frame[:, :, ::-1])
                    # 标定观测区域
                    cv2.rectangle(frame, (self.field_view[0], self.field_view[1]), (self.field_view[2], self.field_view[3]), (100, 255, 125), 3, 1)
                    if out is not None:
                        objs, categorys = out
                        if len(objs) < 1:
                            continue
                        objs, categorys = self._filter_obj(objs, categorys)
                        is_start.extend(zip(objs, categorys))

                        # 可视化结果
                        frame = self._visual(frame, objs=objs, categorys=categorys)
                else:
                    # 计算和上一帧 real_cp 的移动量并更新real_cp
                    x, y, w, h = self.tracker.update(frame)
                    cx = x + w / 2
                    cy = y + h / 2

                    x_offset = cx - real_cp[0] 
                    y_offset = cy - real_cp[1]

                    # 全局调整量
                    up_val += int(x_offset / w * 50)

                    # 更新 real_cp
                    real_cp = [cx, cy]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 0, 100), 2, 1)
                    frame = frame.astype(np.int32)
                    # 跟踪已经启动, 直到视频结束
                    if start_type == 1:
                        # 拳启动，调整 R 通道
                        frame[:, :, 0] += up_val
                    else:
                        # 掌启动，调整 B 通道
                        frame[:, :, 0] += up_val

                frame = np.clip(frame, 0, 255).astype(np.uint8)
                # cv2.imwrite("out.jpg", frame)
                # exit()
                video_writer.write(frame)
                if is_show:
                    cv2.imshow("demo", frame)
                    cv2.waitkey(5)
            except StopIteration as e:
                print('Done!')
                break
        video_writer.release()
