import cv2
import time

if __name__ == '__main__':

    cv2.namedWindow("camera", 1)
    # 开启ip摄像头
    # video = "http://admin:admin@192.168.30.241:8081/"  # 此处@后的ipv4 地址需要修改为自己的地址
    video = "rtsp://admin:admin@192.168.30.241:8554/live/"  # 此处@后的ipv4 地址需要修改为自己的地址
    capture = cv2.VideoCapture(video)

    num = 0
    while True:
        success, img = capture.read()
        cv2.imshow("camera", img)

        # 按键处理，注意，焦点应当在摄像头窗口，不是在终端命令行窗口
        key = cv2.waitKey(1)

        if key == 27:
            # esc键退出
            print("esc break...")
            break   
        if key == ord(' '):
            # 保存一张图像
            num = num+1
            filename = "frames_%s.jpg" % num
            cv2.imwrite(filename, img)

    capture.release()
    cv2.destroyWindow("camera")
