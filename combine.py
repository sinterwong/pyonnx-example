from inference import KeypointsLPN, DetectorYolov5, Classifier
from tools.visualize import visualize
import glob
import cv2
from tqdm import tqdm
from imutils import paths
import numpy as np
import os


class HandDetectionInference(object):
    def __init__(self, model_file, input_size=320, conf_thres=0.3, iou_thres=0.4):
        self._detector = DetectorYolov5(
            model_file, conf_thres=conf_thres, iou_thres=iou_thres, input_size=input_size)

    def single_image(self, frame):
        return self._detector.forward(frame)

    def visual(self, out, frame):
        result = frame.copy()
        for _, dr in enumerate(out):
            cv2.rectangle(result, (dr[0], dr[1]), (dr[2], dr[3]), 255, 2, 1)
        return result

    def crop(self, out, frame):
        image = frame.copy()
        result = []
        for _, dr in enumerate(out):
            result.append(image[dr[1]: dr[3], dr[0]: dr[2], :])
        return result

    def video(self, video_file, save_folder="output", save_result=None, visual_file="out.avi"):
        if save_folder and not os.path.exists(save_folder):
            os.makedirs(save_folder)
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取fps
        # frame_all = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频总帧数
        # video_time = frame_all / fps  # 获取视频总时长

        rval, frame = cap.read()
        h, w, _ = frame.shape
        if visual_file:
            video_writer = cv2.VideoWriter(os.path.join(
                save_folder, visual_file), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        # video_writer = cv2.VideoWriter(os.path.join(save_folder, out_file), cv2.VideoWriter_fourcc(*'AVC1'), fps,(w, h))
        run_count = 0
        e1 = cv2.getTickCount()
        while rval:
            rval, frame = cap.read()
            if frame is not None:
                out = self.single_image(frame[:, :, ::-1])

                if save_result:
                    crop_result = self.crop(out, frame)
                    for i, im in enumerate(crop_result):
                        if 0 in im.shape:
                            continue
                        if not os.path.exists(os.path.join(save_folder, save_result)):
                            os.makedirs(os.path.join(save_folder, save_result))
                        cv2.imwrite(os.path.join(save_folder, save_result, "%d_%03d_%s.jpg" % (
                            run_count, i, os.path.basename(video_file).split(".")[0])), im)
                # 获取单帧图片结果
                if visual_file:
                    visual_result = self.visual(out, frame)
                    video_writer.write(visual_result)
                    # cv2.imwrite("out.jpg", out)
                run_count += 1

        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency()
        # 关闭视频文件
        cap.release()
        video_writer.release()
        print("总耗时：{}s".format(time))
        print("单帧耗时：{}s".format(time / float(run_count)))


class HandOC(object):
    def __init__(self, det_model, cls_model, det_input_size=320, cls_input_size=64, conf_thres=0.5, iou_thres=0.45):
        self._detector = DetectorYolov5(
            det_model, conf_thres=conf_thres, iou_thres=iou_thres, input_size=det_input_size)
        self._classifier = Classifier(cls_model, input_size=cls_input_size)
        # self.idx2classes = dict(enumerate(['0', 'close-back', 'close-front', 'open-back', 'open-front']))
        self.idx2classes = dict(enumerate(['0', 'close', 'open']))

    def single_image(self, frame):
        out = self._detector.forward(frame)
        crop_imgs = self.crop(out, frame)
        reg_out = []
        for i, im in enumerate(crop_imgs):
            if 0 in im.shape or im is None:
                reg_out.append(-1)
                continue
            reg_out.append(self._classifier.forward(im))
        return np.concatenate([out, np.array(reg_out).reshape(-1, 1)], axis=1), crop_imgs

    def visual(self, out, frame):
        result = frame.copy()
        for _, dr in enumerate(out):
            la = self.idx2classes[dr[-1]]
            if dr[-1] == 0:
                continue
            elif dr[-1] == 1:
                # elif dr[-1] == 1 or dr[-1] == 2:
                cv2.rectangle(result, (dr[0], dr[1]),
                              (dr[2], dr[3]), (0, 0, 255), 2, 1)
                cv2.putText(
                    result, la, (dr[0], dr[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            else:
                cv2.rectangle(result, (dr[0], dr[1]),
                              (dr[2], dr[3]), (0, 255, 0), 2, 1)
                cv2.putText(
                    result, la, (dr[0], dr[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        return result

    def crop(self, out, frame):
        image = frame.copy()
        result = []
        for _, dr in enumerate(out):
            result.append(image[dr[1]: dr[3], dr[0]: dr[2], :])
        return result

    def images(self, data_paths, save_result=None, visual_result=None):
        for i, p in tqdm(enumerate(data_paths), total=len(data_paths)):
            frame = cv2.imread(p)
            if 0 in frame.shape or frame is None:
                continue
            out, crop_imgs = self.single_image(frame[:, :, ::-1])

            if save_result:
                for j, im in enumerate(crop_imgs):
                    if 0 in im.shape:
                        continue
                    if not os.path.exists(os.path.join(save_result, self.idx2classes[out[j][-1]])):
                        os.makedirs(os.path.join(save_result, self.idx2classes[out[j][-1]]))
                    cv2.imwrite(os.path.join(save_result, self.idx2classes[out[j][-1]], "%d_%03d_%s.jpg" % (
                        out[j][-1], j, os.path.basename(data_paths[i]).split(".")[0])), im[:, :, ::-1])

    def video(self, video_file, save_folder="output", save_result=None, visual_file="out.avi"):
        if save_folder and not os.path.exists(save_folder):
            os.makedirs(save_folder)
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取fps
        # frame_all = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频总帧数
        # video_time = frame_all / fps  # 获取视频总时长

        rval, frame = cap.read()
        h, w, _ = frame.shape
        if visual_file:
            video_writer = cv2.VideoWriter(os.path.join(
                save_folder, visual_file), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        # video_writer = cv2.VideoWriter(os.path.join(save_folder, out_file), cv2.VideoWriter_fourcc(*'AVC1'), fps,(w, h))
        run_count = 0
        e1 = cv2.getTickCount()
        while rval:
            rval, frame = cap.read()
            if frame is not None:
                out, crop_imgs = self.single_image(frame[:, :, ::-1])
                if save_result:
                    for i, im in enumerate(crop_imgs):
                        if 0 in im.shape:
                            continue
                        if not os.path.exists(os.path.join(save_folder, save_result)):
                            os.makedirs(os.path.join(save_folder, save_result))
                        cv2.imwrite(os.path.join(save_folder, save_result, "%d_%d_%03d_%s.jpg" % (
                            out[i][-1], run_count, i, os.path.basename(video_file).split(".")[0])), im)
                # 获取单帧图片结果
                if visual_file:
                    visual_result = self.visual(out, frame)
                    video_writer.write(visual_result)
                    # cv2.imwrite("out.jpg", visual_result)
                    # exit()
                run_count += 1

        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency()
        # 关闭视频文件
        cap.release()
        video_writer.release()
        print("总耗时：{}s".format(time))
        print("单帧耗时：{}s".format(time / float(run_count)))

