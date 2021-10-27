# pyonnx-example

## introduce

Use python to realize the inference function of the DL model based on the onnxruntime framework. 

Note: Here the models is trained by pytorch v1.6

## requirements 

- onnx == 1.8.1
- onnxruntime == 1.7.0
- onnxruntime-gpu == 1.3.0
- opencv-python == 4.2.0.32

## Run demo

All demos is named in the format main_xxx_.py, You can run demo in the following ways.

```
python main_pose_.py --det_model_path weights/yolov5s.onnx \
			         --pose_model_path weights/hrnet_w32_dark_128x96_trim.onnx \
			         --det_input_size 640 \
			         --pose_input_size (128,96) \
			         --type darkpose or baseline \
			         --video_path data/video/demo.mp4 \
				 --im_path data/det/zidane.jpg \
			         --out_root data/main_result/pose
```


## Model zoo

| model type | link |
| ---- | ---- |
| yolov5s | [提取码:04y8](https://pan.baidu.com/s/1jYgQ_1ZlFxr4idyl-5hXlA)  |
| LPN-coco17  | [提取码:eito](https://pan.baidu.com/s/1RbOjEBbnnplOE5MzyFjyjw) |
| hrnet_dark-wholebody  | [提取码:xiap](https://pan.baidu.com/s/1cJGnoh07M7nwwO8s6x5CBw) |
| hand-pose | [提取码:x9ar](https://pan.baidu.com/s/1wl6B6SI6_kisDyO2THa9Mg) |
| hand-recognition | [提取码:h898](https://pan.baidu.com/s/1wuXDQKKAJK28-PcKF-vJsw) |

## Reference
<https://github.com/ultralytics/yolov5>

<https://github.com/zhang943/lpn-pytorch>

<https://codechina.csdn.net/EricLee/handpose_x>

<https://github.com/open-mmlab/mmpose>

