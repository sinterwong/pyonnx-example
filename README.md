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
- [ ] Gesture recognition combination module added to KCF
- [ ] Based on gesture recognition to achieve up/down page,play,exit four functions
- [ ] Realize action recognition based on continuous frame

## Model zoo

coming soon

## Visualization
### coco-wholebody 133 keypoints(yolov5s + hrnet_w32_dark_128x96)
![133pose_demo](https://github.com/SinterCVer/pyonnx-example/blob/master/data/main_result/pose/133demo.gif?raw=true)

### coco 17 keypoints(yolov5s + lpn_50_256x192)
![17pose_demo](https://github.com/SinterCVer/pyonnx-example/blob/master/data/main_result/pose2/17demo.gif?raw=true)
