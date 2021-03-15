import os
import numpy as np
import cv2
import random
from inference import DetectorYolov5, Classifier
from tracker import KCFtracker
from .base import CombineBase


class GestureRecognition(CombineBase):
    def __init__(self, det_model_file, cls_model_file, det_input_size, cls_input_size, det_conf_thr=0.5, det_iou_thr=0.45, is_tracker=True):

        # load detection model 
        self._detector = DetectorYolov5(
            det_model_file, input_size=det_input_size, conf_thres=det_conf_thr, iou_thres=det_iou_thr)
        
        # load classifier model
        self._classifier = Classifier(cls_model_file, input_size=cls_input_size)

        # idx -> classes
        self.idx2classes = dict(enumerate(['0', 'close', 'open']))

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

    def video_tracker_demo(self, video_file, out_root=None, is_show=False):
        if out_root and not os.path.exists(out_root):
            os.makedirs(out_root)
        frame_iter = self._video(video_file)
        fps, h, w = next(frame_iter)
        video_writer = cv2.VideoWriter(os.path.join(out_root, os.path.basename(video_file)), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

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
                            frame = self._visual(frame, objs=objs, categorys=categorys)
                    else:
                        if objs is not None:
                            x, y, w, h = self.tracker.update(frame)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 0, 100), 2, 1)
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
                        frame = self._visual(frame, objs=objs, categorys=categorys)
                        
                video_writer.write(frame)
                if is_show:
                    cv2.imshow("demo", frame)
                    cv2.waitkey(5)
            except StopIteration as e:
                print('Done!')
                break
        video_writer.release()    

    def grouping(self, boxes, scores, iou_thres, min_group=60):
        groups = []
        tmp_boxes = boxes.copy()
        order = scores.argsort()[::-1]
        areas = self._detector.box_area(boxes)
        while order.size > 1:
            i = order[0]  # 取最大得分

            # 计算最高得分与剩余矩形框的相较区域
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter) + 1e-8

            # 保留小于阈值的 box 继续分组
            inds = np.where(iou < iou_thres)[0]
            keep_inds = np.where(iou > iou_thres)[0]
            if len(keep_inds) >= min_group:
                # 若数量大于 min_group 即可保留
                groups.append(keep_inds)

            # 注意这里索引加了1,因为iou数组的长度比order数组的长度少一个
            order = order[inds + 1]
        return groups

    def _action_page_up(self):
        """ 向上翻页逻辑

        Return:
            result: 0 无翻页动作, 1 激活
        """
        pass
    
    def _action_page_down(self):
        """ 向下翻页逻辑

        Return:
            result: 0 无翻页动作, 1 激活
        """
        pass

    def _action_play(self):
        """ 播放/暂停逻辑

        Return:
            result: 0 无播放/暂停动作, 1 激活
        """
        pass

    def _action_exit(self):
        """ 退出逻辑

        Return:
            result: 0 无退出动作, 1 激活
        """
        pass
    
    def _action_enabled(self, res, iou_thr=0.55, ratio=0.7, cls_id=2):
        """ 判定是否达到启动条件
        Args:
            res: [tuple(det_out, category_out), ...]

        Return:
            region: 如果达成启动条件则返回此区域, 否则为None
        """
        # 所有的目标通过计算 IOU 来分组（60个以上组成组)
        objs = np.vstack([np.concatenate([o, np.array(c).reshape(-1, 1)], axis=1) for o, c in res])
        # TODO 临时去除det_cls
        # objs = np.delete(objs, 5, axis=1)
        boxes, scores = objs[:, :4], objs[:, 4]
        groups = self.grouping(boxes, scores, iou_thr)

        regions = []
        # 过滤 'open' 数量不足阈值的组
        for i, gid in enumerate(groups):
            g = objs[gid]
            print(np.sum(g[:, -1] == cls_id), g.shape)
            print(np.sum(g[:, -1] == cls_id) * 1.0 / g.shape[0])
            # 类别为 open 的数量
            if np.sum(g[:, -1] == cls_id) * 1.0 / g.shape[0] > ratio:
                regions.append(g)
        
        # 如果此时剩余组的数量仍大于1, 则使用位置最高的组
        if len(regions) > 1:
            idx = np.argmin([g[-1][1] for g in regions], axis=0)
            return regions[idx][-1]
        elif len(regions) == 1:
            return regions[0][-1]
        else:
            return None

    def action_demo(self, video_file, out_root=None, is_show=False):
        if out_root and not os.path.exists(out_root):
            os.makedirs(out_root)
        frame_iter = self._video(video_file)
        fps, h, w = next(frame_iter)
        # self._video 之后生成 self.ofps, self.ow, self.oh
        video_writer = cv2.VideoWriter(os.path.join(out_root, os.path.basename(video_file)), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

        # TODO 启动判定参数
        # start_queue = []
        # num_node = 100
        # start_condition = 80
        # valid_count = 0
        # region = None
        count = -1
        objs = None
        categorys = None
        while True:
            # TODO
            # if (valid_count + 1) % num_node == 0:
            #     if len(start_queue) > start_condition:
            #         region = self._action_enabled(start_queue)
            #         print(region)
            #     else:
            #         continue
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
                            frame = self._visual(frame, objs=objs, categorys=categorys)
                    else:
                        if objs is not None:
                            x, y, w, h = self.tracker.update(frame)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 0, 100), 2, 1)
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
                        frame = self._visual(frame, objs=objs, categorys=categorys)
                        
                video_writer.write(frame)
                if is_show:
                    cv2.imshow("demo", frame)
                    cv2.waitkey(5)
            except StopIteration as e:
                print('Done!')
                break
        video_writer.release()

