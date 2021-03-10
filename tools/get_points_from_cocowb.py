from xtcocotools.coco import COCO
import os
import warnings
import numpy as np
import cv2
import pickle as pk
from tqdm import tqdm


class CocoWholeBodyDataset(object):
    """CocoWholeBodyDataset dataset for top-down pose estimation.

    `Whole-Body Human Pose Estimation in the Wild' ECCV'2020
    More details can be found in the `paper
    <https://arxiv.org/abs/2007.11858>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    In total, we have 133 keypoints for wholebody pose estimation.

    COCO-WholeBody keypoint indexes::
        0-16: 17 body keypoints
        17-22: 6 foot keypoints
        23-90: 68 face keypoints
        91-132: 42 hand keypoints

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix):

        self.annotations_path = ann_file
        self.img_prefix = img_prefix
        self.ann_info = {}

        self.ann_info['num_joints'] = 133
        self.ann_info['flip_pairs'] = self._make_flip_pairs()
        self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.ann_info['lower_body_ids'] = (11, 12, 13, 14, 15, 16)
        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.ones(
            (self.ann_info['num_joints'], 1), dtype=np.float32)

        self.body_num = 17
        self.foot_num = 6
        self.face_num = 68
        self.left_hand_num = 21
        self.right_hand_num = 21

        # 'https://github.com/jin-s13/COCO-WholeBody/blob/master/'
        # 'evaluation/myeval_wholebody.py#L170'
        self.sigmas_body = [
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]
        self.sigmas_foot = [0.068, 0.066, 0.066, 0.092, 0.094, 0.094]
        self.sigmas_face = [
            0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031, 0.025, 0.020,
            0.023, 0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045, 0.013,
            0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015,
            0.009, 0.007, 0.007, 0.007, 0.012, 0.009, 0.008, 0.016, 0.010,
            0.017, 0.011, 0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011,
            0.012, 0.010, 0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007,
            0.010, 0.008, 0.009, 0.009, 0.009, 0.007, 0.007, 0.008, 0.011,
            0.008, 0.008, 0.008, 0.01, 0.008
        ]
        self.sigmas_lefthand = [
            0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035,
            0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02,
            0.019, 0.022, 0.031
        ]
        self.sigmas_righthand = [
            0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035,
            0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02,
            0.019, 0.022, 0.031
        ]

        self.sigmas_wholebody = (
            self.sigmas_body + self.sigmas_foot + self.sigmas_face +
            self.sigmas_lefthand + self.sigmas_righthand)

        self.sigmas = np.array(self.sigmas_wholebody)

        self.coco = COCO(ann_file)

        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            (self._class_to_coco_ind[cls], self._class_to_ind[cls])
            for cls in self.classes[1:])
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'coco_wholebody'

        print(f'=> num_images: {self.num_images}')

    @staticmethod
    def _make_flip_pairs():
        body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                [15, 16]]
        foot = [[17, 20], [18, 21], [19, 22]]

        face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

        hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                [111, 132]]

        return body + foot + face + hand

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def get_img_info(self, img_id):
        return self._load_coco_keypoint_annotation_kernel(img_id)

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        bbox_id = 0
        for obj in objs:
            if max(obj['keypoints']) == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            lefthand_joints_3d = np.zeros(
                (self.left_hand_num, 3), dtype=np.float32)
            lefthand_joints_3d_visible = np.zeros(
                (self.left_hand_num, 3), dtype=np.float32)

            righthand_joints_3d = np.zeros(
                (self.right_hand_num, 3), dtype=np.float32)
            righthand_joints_3d_visible = np.zeros(
                (self.right_hand_num, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints'] + obj['foot_kpts'] +
                                 obj['face_kpts'] + obj['lefthand_kpts'] +
                                 obj['righthand_kpts']).reshape(-1, 3)

            lefthand_keypoints = np.array(obj['lefthand_kpts']).reshape(-1, 3)
            righthand_keypoints = np.array(
                obj['righthand_kpts']).reshape(-1, 3)

            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3] > 0)

            lefthand_joints_3d[:, :2] = lefthand_keypoints[:, :2]
            lefthand_joints_3d_visible[:, :2] = np.minimum(
                1, lefthand_keypoints[:, 2:3] > 0)

            righthand_joints_3d[:, :2] = righthand_keypoints[:, :2]
            righthand_joints_3d_visible[:, :2] = np.minimum(
                1, righthand_keypoints[:, 2:3] > 0)

            image_file = os.path.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'lefthand_joints_3d': lefthand_joints_3d,
                'lefthand_joints_3d_visible': lefthand_joints_3d_visible,
                'righthand_joints_3d': righthand_joints_3d,
                'righthand_joints_3d_visible': righthand_joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1

        return rec

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            cuts = np.cumsum([
                0, self.body_num, self.foot_num, self.face_num,
                self.left_hand_num, self.right_hand_num
            ]) * 3

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point[cuts[0]:cuts[1]].tolist(),
                'foot_kpts': key_point[cuts[1]:cuts[2]].tolist(),
                'face_kpts': key_point[cuts[2]:cuts[3]].tolist(),
                'lefthand_kpts': key_point[cuts[3]:cuts[4]].tolist(),
                'righthand_kpts': key_point[cuts[4]:cuts[5]].tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results


def get_result(im_path, hand_joints_3d, joints_3d_visible):
    image = cv2.imread(im_path)
    # bbox = list(map(int, rec['bbox']))
    # person_img = image[bbox[1]: bbox[3], bbox[0]: bbox[2], :]

    # 两个点都大于 0 的情况下参与最大外界矩阵运算
    keep_points = np.logical_or(joints_3d_visible[:, 0], joints_3d_visible[:, 1])

    # if not keep_points.all():
    #     print(hand_joints_3d)
    #     print(keep_points)
    #     exit()
    temp_hand_joints_3d = hand_joints_3d[keep_points]

    # 基于手部关键点获取手部截图
    hand_bbox = cv2.boundingRect(temp_hand_joints_3d[:, 0:2].astype(np.int32))

    if (hand_bbox[2] < 100 or hand_bbox[3] < 100):
        return None, None

    hand_img = image[hand_bbox[1]: hand_bbox[1] + hand_bbox[3], hand_bbox[0]: hand_bbox[0] + hand_bbox[2], :]
    h, w, _ = hand_img.shape
    
    # 回归手部关键点坐标
    hand_joints_3d[:, 0] -= hand_bbox[0]
    hand_joints_3d[:, 1] -= hand_bbox[1]

    # 归一化（仅解决了关键点坐标在 0~1 空间上的相对位置
    hand_joints_3d[:, 0] /= w
    hand_joints_3d[:, 1] /= h

    # 将异常关键点标记
    hand_joints_3d[~keep_points] = -999

    return hand_img, hand_joints_3d


def main():
    out_home = "../data/hand_keypoints"
    out_images_root = os.path.join(out_home, "images")
    if not os.path.exists(out_home):
        os.makedirs(out_home)

    if not os.path.exists(out_images_root):
        os.makedirs(out_images_root)
        
    t = "train"

    coco_home = "/home/wangjq/wangxt/datasets/COCO2017/"
    coco_annotation_home = os.path.join(coco_home, "annotations")

    train_ann_file = os.path.join(
        coco_annotation_home, "coco_wholebody_train_v1.0.json")
    val_ann_file = os.path.join(
        coco_annotation_home, "coco_wholebody_val_v1.0.json")

    train_img_prefix = os.path.join(coco_home, "train2017")
    val_img_prefix = os.path.join(coco_home, "val2017")

    if t != "train":
        datasets = CocoWholeBodyDataset(val_ann_file, val_img_prefix)
    else:
        datasets = CocoWholeBodyDataset(train_ann_file, train_img_prefix)

    left_hand_images, left_hand_names, left_hand_joints, right_hand_images, right_hand_names, right_hand_joints = [], [], [], [], [], []
    for i, idx in tqdm(enumerate(datasets.img_ids), total=len(datasets.img_ids)):
        results = datasets.get_img_info(idx)
        for j, rec in enumerate(results):

            if(np.mean(rec['lefthand_joints_3d']) > 0):
                lefthand_im, lefthand_joints = get_result(rec['image_file'], rec['lefthand_joints_3d'], rec['lefthand_joints_3d_visible'])
                if lefthand_im is None or 0 in lefthand_im.shape:
                    continue
                left_hand_name = "%d_%d_%d_lefthand.jpg" % (i, j, idx)
                left_hand_names.append(left_hand_name)
                left_hand_images.append(lefthand_im)
                left_hand_joints.append(lefthand_joints)

                cv2.imwrite(os.path.join(out_images_root, left_hand_name), lefthand_im)

                # points = [(int(x), int(y)) for x, y in lefthand_joints[:, :2]]
                # for p in points:
                #     if p[0] < -100 or p[1] < -100:
                #         continue
                    # cv2.circle(lefthand_im, p, 1, (0, 0, 255), 0)
                # cv2.imshow("lefthand_im", lefthand_im)
                # cv2.waitKey(5000)

            if(np.mean(rec['righthand_joints_3d']) > 0):
                righthand_im, righthand_joints = get_result(rec['image_file'], rec['righthand_joints_3d'], rec['righthand_joints_3d_visible'])
                if righthand_im is None or 0 in righthand_im.shape:
                    continue
                right_hand_name = "%d_%d_%d_righthand.jpg" % (i, j, idx)
                right_hand_names.append(right_hand_name)
                right_hand_images.append(righthand_im)
                right_hand_joints.append(righthand_joints)

                cv2.imwrite(os.path.join(out_images_root, right_hand_name), righthand_im)

    with open(os.path.join(out_home, "left_hand_images_%s.pkl" % t), 'wb') as wf:
        pk.dump(left_hand_names, wf)
    
    with open(os.path.join(out_home, "left_hand_joints_%s.pkl" % t), 'wb') as wf:
        pk.dump(left_hand_joints, wf)

    with open(os.path.join(out_home, "right_hand_images_%s.pkl" % t), 'wb') as wf:
        pk.dump(right_hand_names, wf)

    with open(os.path.join(out_home, "right_hand_joints_%s.pkl" % t), 'wb') as wf:
        pk.dump(right_hand_joints, wf)


if __name__ == "__main__":
    main()
