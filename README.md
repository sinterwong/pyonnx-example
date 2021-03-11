# pyonnx-example

## introduce

​	Use python to realize the reasoning function of the deep learning model based on the onnxruntime reasoning framework. The onnx model can be converted into most mainstream deep learning inference framework models, so you can test whether the onnx model is correct before deploying the model.

 Note: The model here is trained by pytorch 1.6 and converted  by onnx 1.8.1

## requirements 

- onnx == 1.8.1
- onnxruntime == 1.7.0 or onnxruntime-gpu == 1.3.0
- opencv-python == 4.2.0.32



## Run demo

The demo is named in the format main_xxx_.py, You can run the code with the following example.

```
python main_detector_.py --model_path weights/yolov5s.onnx \
						--im_path data/det/zidane.jpg \
						--input_size 640 \
						--out_root data/main_result/detection
```

```
python main_keypoints_.py --model_path weights/hrnet_w32_dark_256x192.onnx \
						--im_path data/person/004.jpg \
						--out_root data/main_result/keypoints
```

```
python main_classifier_.py --model_path weights/hand-recognition_0.992_3.onnx \
						--im_path data/hand/close.jpg \
						--input_size 64 \
						--out_root data/main_result/classifier
```

## Model zoo

coming soon

## Visualization

![133pose_demo](https://github.com/SinterCVer/pyonnx-example/blob/master/data/main_result/keypoints/004.jpg?raw=true)

![17pose_demo](https://github.com/SinterCVer/pyonnx-example/blob/master/data/main_result/detection/zidane.jpg?raw=true)