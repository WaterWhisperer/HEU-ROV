import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path
import random

model = YOLO('crack/weights/yolov8n.pt')

# 计算检测到的物体中心的平均距离的函数
def get_mid_pos(frame, box, depth_data, randnum):
    distance_list = []

    # 确定深度索引的中心像素位置
    mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
    print(box)
    print(mid_pos)

    # 确定深度搜索范围
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))
    print(min_val)

    for i in range(randnum):
        # 添加随机偏差进行深度采样
        bias = random.randint(-min_val // 4, min_val // 4)

        # 获取索引位置处的深度值
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        print(dist)

        # 在帧上可视化采样点
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255, 0, 0), -1)

        # 考虑有效的深度值并将其添加到列表中
        if dist:
            distance_list.append(dist)

    # 对深度值进行排序并应用中值滤波
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]

    # 返回滤波后距离的平均值
    return np.mean(distance_list)


# 在图像上显示检测到的对象，包括类别标签和估计的距离
def dectshow(org_img, boxs, depth_data):
    img = org_img.copy()

    for box in boxs:
        # 在检测到的物体周围画矩形
        # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # 获取并显示物体的估计距离
        dist = get_mid_pos(org_img, box, depth_data, 24)
        if (dist / 1000) <= 5:
            cv2.putText(img, str('warning:Please keep a safe distance!'),
                        (int(box[0]), int(box[1] - 80)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        cv2.putText(img, str(dist / 1000)[:4] + 'm',
                    (int(box[0]), int(box[1]-40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 显示带注释的图像
    cv2.imshow('dec_img', img)



# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
args.input = r'G:\BAG\20240121_172001.bag'

# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

# Create pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, args.input,repeat_playback=False)

# Configure the pipeline to stream the depth stream
# Change this parameters according to the recorded bag file resolution
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

# config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming from file
pipeline.start(config)

device = pipeline.get_active_profile().get_device()
playback = device.as_playback()
playback.set_real_time(False)

# Create colorizer object
colorizer = rs.colorizer()

# Streaming loop
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 定义视频编解码器
# frame_id = 0
# out = cv2.VideoWriter('output.mp4', fourcc, 30, (1280, 720))  # 创建VideoWriter对象
try:
    while True:

        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # Colorize depth frame to jet colormap
        # depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_frame.get_data())

        results = model(color_image)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes.xyxy.cpu().tolist()


        # 显示检测到的对象和估计的距离
        dectshow(annotated_frame, boxes, depth_color_image)

        # # 在深度图像上应用颜色映射
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #
        # # 水平堆叠彩色和深度图像以进行显示
        # images = np.hstack((color_image, depth_colormap))

        # 显示组合图像
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', annotated_frame)

        # 等待按键，按 'q' 或 'ESC' 键退出
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        # out.write(color_image)
        # if pressed escape exit program
finally:

        cv2.destroyAllWindows()
        pipeline.stop()

