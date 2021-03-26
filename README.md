# pyonnx-example

## introduce

​	Use python to realize the inference function of the deep learning model based on the onnxruntime reasoning framework. The onnx model can be converted into most mainstream deep learning inference framework models, so you can test whether the onnx model is correct before deploying the model.

 Note: The model here is trained by pytorch 1.6 and converted  by onnx 1.8.1

## requirements 

- onnx == 1.8.1
- onnxruntime == 1.7.0 or onnxruntime-gpu == 1.3.0
- opencv-python == 4.2.0.32

## Run demo

The demo is named in the format main_xxx_.py, You can run the code with the following example.

```
python main_pose_.py --det_model_path weights/yolov5s.onnx \
			         --pose_model_path data/det/zidane.jpg \
			         --det_input_size 640 \
			         --pose_input_size (128,96) \
			         --type darkpose or baseline \
			         --video_path data/video/demo.mp4 \
				 --im_path data/det/zidane.jpg \
			         --out_root data/main_result/pose
```

## TODO
- [x] Gesture recognition combination module added to KCF
- [ ] Based on gesture recognition to achieve up/down page,play,exit four functions
- [ ] Realize action recognition based on continuous frame

## Model zoo

| model type | link |
| ---- | ---- |
| yolov5s | [提取码:04y8](https://pan.baidu.com/s/1jYgQ_1ZlFxr4idyl-5hXlA)  |
| LPN-coco17  | [提取码:eito](https://pan.baidu.com/s/1RbOjEBbnnplOE5MzyFjyjw) |
| hrnet_dark-wholebody  | [提取码:xiap](https://pan.baidu.com/s/1cJGnoh07M7nwwO8s6x5CBw) |
| hand-pose | [提取码:x9ar](https://pan.baidu.com/s/1wl6B6SI6_kisDyO2THa9Mg) |
| hand-recognition | [提取码:h898](https://pan.baidu.com/s/1wuXDQKKAJK28-PcKF-vJsw) |

## Visualization
### coco-wholebody 133 keypoints(yolov5s + hrnet_w32_dark_128x96)
![133pose_demo](https://github.com/SinterCVer/pyonnx-example/blob/master/data/main_result/pose/133demo.gif?raw=true)

### coco 17 keypoints(yolov5s + lpn_50_256x192)
![17pose_demo](https://github.com/SinterCVer/pyonnx-example/blob/master/data/main_result/pose2/17demo.gif?raw=true)

### hand sliding motion(yolov5s + resnet18 + kcf)
![slide_demo](https://github.com/SinterCVer/pyonnx-example/blob/master/data/main_result/gesture/fist_move.gif?raw=true)

### hand gestures(yolov5s + res50_hand_pose)
![gesture_demo](https://github.com/SinterCVer/pyonnx-example/blob/master/data/main_result/gesture2/gesture.gif?raw=true)

## Reference
<https://github.com/ultralytics/yolov5>

<https://github.com/zhang943/lpn-pytorch>

<https://codechina.csdn.net/EricLee/handpose_x>

<https://github.com/open-mmlab/mmpose>

