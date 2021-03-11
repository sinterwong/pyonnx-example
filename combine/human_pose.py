import os
import numpy as np
import cv2
import random
from inference import DetectorYolov5, KeypointsDarkPose, KeypointsLPN
from .base import CombineBase


class HumanPose(CombineBase):
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
            raise Exception("HumanPose init failed: Unsupported type {}".format(pose_type))
        self.pose_thr = pose_thr
    
    def _single_frame(self, img):
        out = self._detector.forward(img)
        if out.shape[0] < 1:
            return None
        points = []
        objs = []
        for _, dr in enumerate(out):
            if int(dr[-1]) != 0:
                continue
            person_image = img[dr[1]: dr[3], dr[0]: dr[2], :]
            if person_image.shape[0] < 10 or person_image.shape[1] < 10:
                continue
            point = self._pose_detector.forward(person_image)
            # points reg to src image
            point[:, 0] += dr[0]
            point[:, 1] += dr[1]
            points.append(point)
            objs.append(dr)
        return objs, points

    def _draw_skeleton_17(self, aa, kp, show_skeleton_labels=False):
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 
        'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 
        'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

        for i, j in skeleton:
            if kp[i-1][0] >= 0 and kp[i-1][1] >= 0 and kp[j-1][0] >= 0 and kp[j-1][1] >= 0 and \
                (len(kp[i-1]) <= 2 or (len(kp[i-1]) > 2 and  kp[i-1][2] > 0.1 and kp[j-1][2] > 0.1)):
                cv2.line(aa, tuple(kp[i-1][:2]), tuple(kp[j-1][:2]), (0,255,255), 2)
        for j in range(len(kp)):
            if kp[j][0] >= 0 and kp[j][1] >= 0:

                if len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 1.1):
                    cv2.circle(aa, tuple(kp[j][:2]), 2, tuple((0,0,255)), 2)
                elif len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1):
                    cv2.circle(aa, tuple(kp[j][:2]), 2, tuple((255,0,0)), 2)

                if show_skeleton_labels and (len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1)):
                    cv2.putText(aa, kp_names[j], tuple(kp[j][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))

    def visualize_pose17(self, img, keypoints, show_skeleton_labels=False, thresh=0.2):

        color_map = [(0, 0, 0), (0, 255, 0), (255, 128, 0), (255, 255, 0), (255, 0, 255), (255, 128, 255), (128, 255, 128), (128, 255, 255), (255, 255, 128), (0, 128, 255), (0, 255, 128),
                (255, 0, 128), (0, 215, 255), (255, 0, 255), (255, 128, 0), (128, 128, 255), (0, 255, 255), (0, 69, 255), (0, 69, 255), (255, 204, 204), (204, 255, 255)]

        im = np.array(img).astype(np.uint8)
        colors = dict()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(len(keypoints)):
            self._draw_skeleton_17(im, keypoints[i], show_skeleton_labels)

        return im

    def visualize_pose133(self, img, keypoints, thresh=0.2):
        img_h, img_w, _ = img.shape
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [16, 18],
                    [16, 19], [16, 20], [17, 21], [17, 22], [17, 23], [92, 93],
                    [93, 94], [94, 95], [95, 96], [92, 97], [97, 98], [98, 99],
                    [99, 100], [92, 101], [101, 102], [102, 103], [103, 104],
                    [92, 105], [105, 106], [106, 107], [107, 108], [92, 109],
                    [109, 110], [110, 111], [111, 112], [113, 114], [114, 115],
                    [115, 116], [116, 117], [113, 118], [118, 119], [119, 120],
                    [120, 121], [113, 122], [122, 123], [123, 124], [124, 125],
                    [113, 126], [126, 127], [127, 128], [128, 129], [113, 130],
                    [130, 131], [131, 132], [132, 133]]

        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

        pose_limb_color = palette[
            [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16] +
            [16, 16, 16, 16, 16, 16] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ]]
        pose_kpt_color = palette[
            [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
            [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]
        radius = 1

        for _, kpts in enumerate(keypoints):
            # draw each point on image
            if pose_kpt_color is not None:
                assert len(pose_kpt_color) == len(kpts)
                for kid, kpt in enumerate(kpts):
                    x_coord, y_coord, kpt_score = int(kpt[0]), int(
                        kpt[1]), kpt[2]
                    if kpt_score > thresh:
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(img, (int(x_coord), int(y_coord)),
                                    radius, (int(r), int(g), int(b)),
                                    -1)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                                  1]))
                        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                                  1]))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1, 2] > thresh
                                and kpts[sk[1] - 1, 2] > thresh):
                            r, g, b = pose_limb_color[sk_id]
                            cv2.line(
                                img,
                                pos1,
                                pos2, (int(r), int(g), int(b)),
                                thickness=1)

        return img

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
                out = self._single_frame(frame[:, :, ::-1])
                if out is not None:
                    objs, points = out
                    if len(objs) < 1:
                        continue
                    if len(points[0]) == 17:
                        frame = self.visualize_pose17(frame, np.array(points), show_skeleton_labels=True, thresh=self.pose_thr)
                    elif len(points[0]) == 133:
                        frame = self.visualize_pose133(frame, np.array(points), thresh=self.pose_thr)
                    else:
                        raise Exception("Visualization of only 17 points and 133 points is currently supported")
                video_writer.write(frame)
                if is_show:
                    cv2.imshow("demo", frame)
                    cv2.waitkey(5)
            except StopIteration as e:
                print('Done!')
                break
        video_writer.release()
