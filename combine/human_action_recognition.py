import os
import numpy as np
import cv2
import random
from .human_pose import HumanPose
from inference import DetectorYolov5, KeypointsDarkPose, KeypointsLPN


class HumanActionRecognition(HumanPose):

    def __init__(self, det_model_file, pose_model_file, det_input_size, pose_input_size, pose_type='darkpose', det_conf_thr=0.5, det_iou_thr=0.45, pose_thr=0.25):

        # load detection model
        self._detector = DetectorYolov5(
            det_model_file, input_size=det_input_size, conf_thres=det_conf_thr, iou_thres=det_iou_thr)

        # load pose model, 不同的类型仅后处理的策略不同
        if pose_type == "baseline":
            self._pose_detector = KeypointsLPN(
                pose_model_file, input_size=pose_input_size, conf_thres=pose_thr)
        elif pose_type == "darkpose":
            self._pose_detector = KeypointsDarkPose(
                pose_model_file, input_size=pose_input_size, conf_thres=pose_thr)
        else:
            raise Exception(
                "HumanPose init failed: Unsupported type {}".format(pose_type))
        self.pose_thr = pose_thr

        self.body = [[1, 2], [3, 4], [5, 6], [7, 8],
                     [9, 10], [11, 12], [13, 14], [15, 16]]
        self.foot = [[17, 20], [18, 21], [19, 22]]
        self.face = [[23, 39], [24, 38], [25, 37], [26, 36],
                     [27, 35], [28, 34], [29, 33], [30, 32],
                     [40, 49], [41, 48], [42, 47], [43, 46],
                     [44, 45], [54, 58], [55, 57], [59, 68],
                     [60, 67], [61, 66], [62, 65], [63, 70],
                     [64, 69], [71, 77], [72, 76], [73, 75],
                     [78, 82], [79, 81], [83, 87], [84, 86],
                     [88, 90]]
        self.hand = [[91, 112], [92, 113], [93, 114], [94, 115],
                     [95, 116], [96, 117], [97, 118], [98, 119],
                     [99, 120], [100, 121], [101, 122], [102, 123],
                     [103, 124], [104, 125], [105, 126], [106, 127],
                     [107, 128], [108, 129], [109, 130], [110, 131],
                     [111, 132]]

        self.action_dict = {
            "0000": "fist",
            "1000": "one",
            "0100": "fuck",
            "1100": "Yeah",
            "1110": "three",
            "1111": "palm",
            "0111": "OK",
            "0001": "low"
        }

    def _cosangle(self, v1, v2):
        """ 计算向量角度
        Args:
            v1:
            v2:
        Return: included_angle(int) 向量角度
        """
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]

        angle1 = np.arctan2(dy1, dx1)
        angle1 = int(angle1 * 180 / np.pi)
        angle2 = np.arctan2(dy2, dx2)
        angle2 = int(angle2 * 180 / np.pi)

        if angle1*angle2 >= 0:
            included_angle = abs(angle1-angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle

    def _finger_status(self, points):
        """ 每个手指状态
        Args:
            points(ndarray): 21 手部关键点结果
        Return:
            cos_angles(list(int)): 向量角度
        """
        # 6(食指), 10(中指), 14(无名指), 18(小拇指) 手指中心
        center_points = points[[6, 10, 14, 18]][:, :2]

        # 手腕坐标
        root_point = points[0][:2]

        # 获取根部到各个指中的向量
        vet_center_points = np.array(
            [np.concatenate([root_point, p], axis=0) for p in center_points])

        # 指间向量获取  8(食指), 12(中指), 16(无名指), 20(小拇指)
        first_points = points[[8, 12, 16, 20]][:, :2]
        second_points = points[[7, 11, 15, 19]][:, :2]
        vet_fingertip_points = np.concatenate(
            [second_points, first_points], axis=1)

        # 计算 cos angle
        cos_angles = np.array([self._cosangle(c, t) for c, t in zip(
            vet_center_points, vet_fingertip_points)])

        # print(cos_angles)
        
        return cos_angles

    def finger_status(self, points, p_thr=0.5, a_thr=90):
        """ 获取除大拇指以外手指状态识别（顺序为 食指, 中指, 无名指, 小拇指）
        Args:
            points(ndarray): 检测的关键点结果
            p_thr(float): 手部关键点参与计算的阈值
            a_thr(int): 向量角度阈值, 大于阈值为合, 小于阈值为开
        Return:
            (list(bool), bbox): 左手每只手指的开合状态和手部框 (ps: True 代表开, False 代表合)
            (list(bool), bbox): 右手每只手指的开合状态和手部框
        """
        lh_status, rh_status, lh_bbox, rh_bbox = [None] * 4
        # 获取手部关键点
        hand_ids = np.array(self.hand)
        lh_ids = hand_ids[:, 0]
        rh_ids = hand_ids[:, 1]

        lh_points = points[lh_ids]
        rh_points = points[rh_ids]

        if np.mean(lh_points[:, -1]) > 0.5:
            lh_status = self._finger_status(lh_points) < a_thr
            lh_bbox = cv2.boundingRect(lh_points[:, :2].astype(np.int32))

        if np.mean(rh_points[:, -1]) > 0.5:
            rh_status = self._finger_status(rh_points) < a_thr
            rh_bbox = cv2.boundingRect(rh_points[:, :2].astype(np.int32))

        return (lh_status, lh_bbox), (rh_status, rh_bbox)

    def hand_action(self, status):
        """ 根据手指的开合状态判断手部动作
        Args:
            status(list(bool)): 四个手指（食指, 中指, 无名指, 小拇指）的状态, True 为开, False 为合
        Return:
            action(str):
                explain: 
                    status = [False, False, False, False]  握拳(fist)
                    status = [True, False, False, False]   一(one)
                    status = [False, True, False, False]   竖中指(fuck)
                    status = [True, True, False, False]    耶(Yeah)
                    status = [True, True, True, False]     三(three))
                    status = [True, True, True, True]      掌(palm)
                    status = [False, True, True, True]     OK(OK)
                    status = [False, False, False, True]   鄙视(low)
        """
        # print("".join(list(map(str, list(map(int, status))))))
        action_encode = "".join(list(map(str, list(map(int, status)))))
        if action_encode in self.action_dict.keys():
            return self.action_dict[action_encode]
        else:
            return "nonsupport"

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
                    # objs list 行人的框信息, points list 每个人的关键点信息
                    objs, points = out
                    if len(objs) < 1:
                        continue
                    # 可视化结果
                    frame = self._visual(frame, points=points)
                    for p in points:
                        # 获取除大拇指以外其余手指的状态
                        hand_info = self.finger_status(np.array(p))
                        # 根据手指状态判断手部动作
                        for hi in hand_info:
                            if hi[0] is not None:
                                status, bbox = hi
                                action = self.hand_action(status)
                                cv2.rectangle( frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1, 1)
                                cv2.putText(frame, action, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
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
            # objs list 行人的框信息, points list 每个人的关键点信息
            (objs, points), frame = out
            if len(objs) < 1:
                return None
            frame = self._visual(frame, objs=objs, points=points)

            for p in points:
                # 获取除大拇指以外其余手指的状态
                hand_info = self.finger_status(np.array(p))
                # 根据手指状态判断手部动作
                for hi in hand_info:
                    if hi[0] is not None:
                        status, bbox = hi
                        action = self.hand_action(status)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1, 1)
                        cv2.putText(frame, action, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        else:
            return None
        if is_show:
            cv2.imshow("demo", frame)
            cv2.waitkey(5000)

        if out_root:
            cv2.imwrite(os.path.join(out_root, os.path.basename(path)), frame)
