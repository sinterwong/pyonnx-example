import os
import numpy as np
import cv2
import random
from inference import DetectorYolov5, KeypointsReg
from tracker import KCFtracker
from .base import CombineBase


class GestureRecognitionPose(CombineBase):
    def __init__(self, det_model_file, pose_model_file, det_input_size, pose_input_size=(256, 256), angle_thr=90, pose_type='regression', det_conf_thr=0.5, det_iou_thr=0.25):

        # load detection model
        self._detector = DetectorYolov5(
            det_model_file, input_size=det_input_size, conf_thres=det_conf_thr, iou_thres=det_iou_thr)

        # load pose model, 不同的类型仅后处理的策略不同
        if pose_type == "regression":
            self._pose_detector = KeypointsReg(
                pose_model_file, input_size=pose_input_size)
        else:
            raise Exception(
                "HumanPose init failed: Unsupported type {}".format(pose_type))

        # 含大拇指
        self.action_dict1 = {
            "00000": "fist",
            "01000": "one",
            "00100": "fuck",
            "11001": "love you",
            "01100": "Yeah",
            "01110": "three",
            "01111": "four",
            "11111": "palm",
            "00111": "OK",
            "00001": "low",
            "11000": "gun",
            "10001": "six",
            "10000": "np"
        }

        # 不含大拇指
        self.action_dict2 = {
            "0000": "fist",
            "1000": "one",
            "0100": "fuck",
            "1001": "love you",
            "1100": "Yeah",
            "1110": "three",
            "1111": "palm",
            "0111": "OK",
            "0001": "low",
        }

        self.a_thr = angle_thr

    def _single_frame(self, img):
        out = self._detector.forward(img)
        if out.shape[0] < 1:
            return None
        points = []
        objs = []
        for _, dr in enumerate(out):
            if int(dr[-1]) != 0:
                continue
            x_min, y_min, x_max, y_max, _, _ = dr
            w_ = max(abs(x_max-x_min), abs(y_max-y_min))
            w_ = w_*1.1
            x_mid = (x_max+x_min)/2
            y_mid = (y_max+y_min)/2
            x1, y1, x2, y2 = int(x_mid-w_/2), int(y_mid - w_/2), int(x_mid+w_/2), int(y_mid+w_/2)
            x1 = np.clip(x1, 0, img.shape[1]-1)
            x2 = np.clip(x2, 0, img.shape[1]-1)
            y1 = np.clip(y1, 0, img.shape[0]-1)
            y2 = np.clip(y2, 0, img.shape[0]-1)

            hand_image = img[y1: y2, x1: x2, :]
            if hand_image.shape[0] < 10 or hand_image.shape[1] < 10:
                continue
            point = self._pose_detector.forward(hand_image[:, :, ::-1])
            # points reg to src image
            point[:, 0] += x1
            point[:, 1] += y1
            points.append(point)
            objs.append(dr)
        return objs, points

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
        # 2(大拇指), 6(食指), 10(中指), 14(无名指), 18(小拇指) 手指中心
        center_points = points[[2, 6, 10, 14, 18]][:, :2]

        # 手腕坐标
        root_point = points[0][:2]

        # 获取根部到各个指中的向量
        vet_center_points = np.array(
            [np.concatenate([root_point, p], axis=0) for p in center_points])

        # 指间向量获取  4(大拇指), 8(食指), 12(中指), 16(无名指), 20(小拇指)
        first_points = points[[4, 8, 12, 16, 20]][:, :2]
        second_points = points[[3, 7, 11, 15, 19]][:, :2]
        vet_fingertip_points = np.concatenate(
            [second_points, first_points], axis=1)

        # 计算 cos angle
        cos_angles = np.array([self._cosangle(c, t) for c, t in zip(
            vet_center_points, vet_fingertip_points)])

        # print(cos_angles)

        return cos_angles

    def finger_status(self, points, a_thr=90):
        """ 获取除大拇指以外手指状态识别（顺序为 食指, 中指, 无名指, 小拇指）
        Args:
            points(ndarray): 检测的关键点结果
            a_thr(int): 向量角度阈值, 大于阈值为合, 小于阈值为开
        Return:
            status(list(bool)): 右手每只手指的开合状态和手部框
        """
        status = self._finger_status(points) < a_thr
        return status

    def hand_action(self, status, pollex=False):
        """ 根据手指的开合状态判断手部动作
        Args:
            status(list(bool)): 四个手指（食指, 中指, 无名指, 小拇指）的状态, True 为开, False 为合
        Return:
            action(str):
                explain（包含大拇指）: 
                    status = [False, False, False, False, False]  握拳(fist)
                    status = [False, True, False, False, False]   一(one)
                    status = [False, False, True, False, False]   竖中指(fuck)
                    status = [True, True, False, False, True]     爱你(IOU)
                    status = [False, True, True, False, False]    耶(Yeah)
                    status = [False, True, True, True, False]     三(three)
                    status = [False, True, True, True, True]      四(Four)
                    status = [True, True, True, True, True]       掌(palm)
                    status = [False, False, True, True, True]     OK(OK)
                    status = [False, False, False, False, True]   鄙视(low)
                    status = [True, True, False, False, False]    手枪(gun)
                    status = [True, False, False, False, True]    六(six)
                    status = [True, False, False, False, True]    赞(np)

                explain（不包含大拇指）: 
                    status = [False, False, False, False]   握拳(fist)
                    status = [True, False, False, False]    一(one)
                    status = [False, True, False, False]    竖中指(fuck)
                    status = [True, False, False, True]     爱你(IOU)
                    status = [True, True, False, False]     耶(Yeah)
                    status = [True, True, True, False]      三(three)
                    status = [True, True, True, True]       掌(palm)
                    status = [False, True, True, True]      OK(OK)
                    status = [False, False, False, True]    鄙视(low)
        """
        # print("".join(list(map(str, list(map(int, status))))))
        action_encode = "".join(list(map(str, list(map(int, status)))))
        try:
            if pollex:
                return self.action_dict1[action_encode]
            else:
                action_encode = action_encode[1:]
                return self.action_dict2[action_encode]
        except Exception as e:
            return "nonsupport"

    def visualize_hand_pose(self, frame, hand_, x, y):
        thick = 2
        colors = [(0, 215, 255), (255, 115, 55),
                  (5, 255, 55), (25, 15, 255), (225, 15, 55)]

        cv2.line(frame, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),
                 (int(hand_['1']['x']+x), int(hand_['1']['y']+y)), colors[0], thick)
        cv2.line(frame, (int(hand_['1']['x']+x), int(hand_['1']['y']+y)),
                 (int(hand_['2']['x']+x), int(hand_['2']['y']+y)), colors[0], thick)
        cv2.line(frame, (int(hand_['2']['x']+x), int(hand_['2']['y']+y)),
                 (int(hand_['3']['x']+x), int(hand_['3']['y']+y)), colors[0], thick)
        cv2.line(frame, (int(hand_['3']['x']+x), int(hand_['3']['y']+y)),
                 (int(hand_['4']['x']+x), int(hand_['4']['y']+y)), colors[0], thick)

        cv2.line(frame, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),
                 (int(hand_['5']['x']+x), int(hand_['5']['y']+y)), colors[1], thick)
        cv2.line(frame, (int(hand_['5']['x']+x), int(hand_['5']['y']+y)),
                 (int(hand_['6']['x']+x), int(hand_['6']['y']+y)), colors[1], thick)
        cv2.line(frame, (int(hand_['6']['x']+x), int(hand_['6']['y']+y)),
                 (int(hand_['7']['x']+x), int(hand_['7']['y']+y)), colors[1], thick)
        cv2.line(frame, (int(hand_['7']['x']+x), int(hand_['7']['y']+y)),
                 (int(hand_['8']['x']+x), int(hand_['8']['y']+y)), colors[1], thick)

        cv2.line(frame, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),
                 (int(hand_['9']['x']+x), int(hand_['9']['y']+y)), colors[2], thick)
        cv2.line(frame, (int(hand_['9']['x']+x), int(hand_['9']['y']+y)),
                 (int(hand_['10']['x']+x), int(hand_['10']['y']+y)), colors[2], thick)
        cv2.line(frame, (int(hand_['10']['x']+x), int(hand_['10']['y']+y)),
                 (int(hand_['11']['x']+x), int(hand_['11']['y']+y)), colors[2], thick)
        cv2.line(frame, (int(hand_['11']['x']+x), int(hand_['11']['y']+y)),
                 (int(hand_['12']['x']+x), int(hand_['12']['y']+y)), colors[2], thick)

        cv2.line(frame, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),
                 (int(hand_['13']['x']+x), int(hand_['13']['y']+y)), colors[3], thick)
        cv2.line(frame, (int(hand_['13']['x']+x), int(hand_['13']['y']+y)),
                 (int(hand_['14']['x']+x), int(hand_['14']['y']+y)), colors[3], thick)
        cv2.line(frame, (int(hand_['14']['x']+x), int(hand_['14']['y']+y)),
                 (int(hand_['15']['x']+x), int(hand_['15']['y']+y)), colors[3], thick)
        cv2.line(frame, (int(hand_['15']['x']+x), int(hand_['15']['y']+y)),
                 (int(hand_['16']['x']+x), int(hand_['16']['y']+y)), colors[3], thick)

        cv2.line(frame, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),
                 (int(hand_['17']['x']+x), int(hand_['17']['y']+y)), colors[4], thick)
        cv2.line(frame, (int(hand_['17']['x']+x), int(hand_['17']['y']+y)),
                 (int(hand_['18']['x']+x), int(hand_['18']['y']+y)), colors[4], thick)
        cv2.line(frame, (int(hand_['18']['x']+x), int(hand_['18']['y']+y)),
                 (int(hand_['19']['x']+x), int(hand_['19']['y']+y)), colors[4], thick)
        cv2.line(frame, (int(hand_['19']['x']+x), int(hand_['19']['y']+y)),
                 (int(hand_['20']['x']+x), int(hand_['20']['y']+y)), colors[4], thick)

    def _visual(self, frame, objs=None, points=None):
        if objs:
            for dr in objs:
                cv2.rectangle(frame, (dr[0], dr[1]),
                              (dr[2], dr[3]), (0, 255, 0), 1, 1)
        if points:
            for pt in points:
                pts_hand = {}  # 构建关键点连线可视化结构
                for i, p in enumerate(pt):
                    cv2.circle(frame, (int(p[0]), int(
                        p[1])), 3, (255, 50, 60), -1)
                    cv2.circle(frame, (int(p[0]), int(
                        p[1])), 1, (255, 150, 180), -1)
                    pts_hand[str(i)] = {}
                    pts_hand[str(i)] = {
                        "x": p[0],
                        "y": p[1],
                    }
                self.visualize_hand_pose(frame, pts_hand, 0, 0)
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
                    # objs list 行人的框信息, points list 每个人的关键点信息
                    objs, points = out
                    if len(objs) < 1:
                        continue
                    # 可视化结果
                    frame = self._visual(frame, objs=objs, points=points)
                    for pi, p in enumerate(points):
                        # 获取除大拇指以外其余手指的状态
                        status = self.finger_status(np.array(p), a_thr=self.a_thr)
                        action = self.hand_action(status, True)
                        cv2.putText(frame, action, (objs[pi][0], objs[pi][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
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
            for pi, p in enumerate(points):
                # 获取除大拇指以外其余手指的状态
                status = self.finger_status(np.array(p), a_thr=self.a_thr)
                action = self.hand_action(status)
                cv2.putText(frame, action, (objs[pi][0], objs[pi][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        else:
            return None
        if is_show:
            cv2.imshow("demo", frame)
            cv2.waitkey(5000)

        if out_root:
            cv2.imwrite(os.path.join(out_root, os.path.basename(path)), frame)
