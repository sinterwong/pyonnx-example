from combine import HandDetectionInference, HandOC
import os
import glob
from imutils import paths
from tqdm import tqdm


def get_from_video(model_path, video_folder, output, input_size=640):
    """ 处理带有类别信息的视频 """
    inference = HandDetectionInference(
        model_path, conf_thres=0.45, input_size=input_size)
    video_paths = glob.glob(video_folder + "/*/*.mp4")
    print("视频数量：", len(video_paths))
    print(video_paths)
    for _, i in tqdm(enumerate(video_paths), total=len(video_paths)):
        save_result = i.split("/")[-2]
        inference.video(i, output, save_result, os.path.basename(i))


def get_from_predict(det_model, cls_model, data_root, output_folder, input_size=[640, 64]):
    data_paths = list(paths.list_images(basePath=data_root))
    inference = HandOC(det_model, cls_model,
                       input_size[0], input_size[1], conf_thres=0.4, iou_thres=0.45)
    inference.images(data_paths, output_folder)


if __name__ == "__main__":
    # # 通过已知的视频类别获取
    model = "weights/hand-yolov5-640.onnx"
    video_file = "/home/wangjq/liuzq/data/tmp"
    output_folder = "/home/wangjq/liuzq/data/tmp/tmp"
    get_from_video(model, video_file, output_folder, 640)

    # 通过手势识别预测刷得数据
    # det_model = "weights/hand-yolov5-320.onnx"
    # cls_model = "weights/hand-recognition_0.992_3.onnx"
    # # data_root = "/home/wangjq/wangxt/datasets/egohands_data/_LABELLED_SAMPLES"
    # data_root = "/home/wangjq/wangxt/datasets/hand_dataset/images"
    # output_folder = "data/hand/results"
    # get_from_predict(det_model, cls_model, data_root,
    #                  output_folder, input_size=[320, 64])
