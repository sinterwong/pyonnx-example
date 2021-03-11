import glob
import cv2
from tqdm import tqdm
from imutils import paths
import numpy as np
import os
import abc
import logging


class CombineBase(object):
    @abc.abstractmethod
    def __init__(self, model_path, thres):
        """Init CombineBase.

        Provide the models path, models initialization parameters and other
        information to load the models.

        Args:
            model_path: Path of the model.
            thres: The threshold value of the algorithm.

        Returns:
            None
        """
        pass

    def __str__(self):
        return super().__str__()
    
    @abc.abstractmethod
    def _single_frame(self, img):
        """Single frame image processing.

        The information of single image processing is obtained by combination algorithm

        Args:
            img: [h, w, c]

        Returns:
            All the information given by the algorithm
        """
        pass
    
    @abc.abstractmethod
    def _visual_single(self, out, frame):
        """Visualize a single image.

        Args:
            out: Information provided by the algorithm.
            frame: The image to be drawn.

        Returns:
            The drawn image.
        """
        pass

    def _crop(self, out, frame):
        """Crop detected objects.

        Args:
            out: Information provided by the detection algorithm.
            frame: The image to be crop.

        Returns:
            result: Object images list
        """
        image = frame.copy()
        result = []
        for _, dr in enumerate(out):
            result.append(image[dr[1]: dr[3], dr[0]: dr[2], :])
        return result

    def _imread(self, path):
        """image read

        Returns:
            result: rgb image [h, w, c]
        """
        if not os.path.exists(path):
            logging.warning("src file not exist: {}".format(path))
            return None
        frame = cv2.imread(path)
        if 0 in frame.shape or frame is None:
            logging.warning("The dimension of the image is wrong: {}".format(frame.shape))
            return None
        return frame[:, :, ::-1]

    def single_image(self, path):
        """single image processing

        Args:
            path: The path of the images.

        Returns:
            result: Object images list
        """
        frame = self._imread(path)
        if frame is None:
            return None
        return self._single_frame(frame)

    def mult_images(self, data_paths):
        """Crop detected objects.

        Args:
            data_paths: The path of the images.

        Returns:
            result: Object images list
        """
        outs = []
        for i, p in tqdm(enumerate(data_paths), total=len(data_paths)):
            outs.append(self.single_image(p))

    def _video(self, video_file):
        print(video_file)
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取fps
        # frame_all = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频总帧数
        # video_time = frame_all / fps  # 获取视频总时长
        rval, frame = cap.read()
        h, w, _ = frame.shape
        yield fps, h, w  # 第一次使用 next 时返回（用于写入视频时的信息）
        while rval:
            rval, frame = cap.read()
            if frame is not None:
                yield frame

        # 关闭视频文件
        cap.release()
