import netron

modelPath = "../weights/handv3-yolov5-320.onnx"
netron.start(modelPath)


# ./MNNConvert -f ONNX --modelFile /home/wangjq/wangxt/workspace/pyonnx-example/weights/hrnet_w32_dark_256x192.onnx --MNNModel /home/wangjq/wangxt/workspace/pyonnx-example/weights/hrnet_w32_dark_256x192.mnn  --bizCode biz
