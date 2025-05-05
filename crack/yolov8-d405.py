import pyrealsense2 as rs
import numpy as np
import cv2
import random
import math
import torch
import time
import sys
from ultralytics import YOLO

# 载入 YOLOv8 模型
model = YOLO('crack/weight/yolov8n.pt')

# -------------------------- 新增部分：鼠标回调相关全局变量 --------------------------
measure_points = []  # 存储测量点的坐标
depth_intrinsics = None  # 存储深度相机内参

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global measure_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(measure_points) < 2:
            measure_points.append((x, y))
        else:
            measure_points = [(x, y)]  # 超过两点时重置

# -----------------------------------------------------------------------------

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
        dist_corrected = dist / 10 # 修正深度值单位为毫米

        # 在帧上可视化采样点
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255, 0, 0), -1)
        print(dist_corrected)

        # 考虑有效的深度值并将其添加到列表中
        if dist_corrected:
            distance_list.append(dist_corrected)

    print('-------------------------')

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
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # 获取并显示物体的估计距离
        dist = get_mid_pos(org_img, box, depth_data, 24)
        print(dist)
        cv2.putText(img, f"{dist:.1f}mm",
                    (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 显示带注释的图像
    cv2.imshow('dec_img', img)


if __name__ == "__main__":
    # -------------------------- 新增部分：获取相机内参 --------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    profile = pipeline.start(config)
    
    # 获取深度流内参
    depth_profile = profile.get_stream(rs.stream.depth)
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
    
    # 创建测量窗口并设置回调
    cv2.namedWindow('Measure')
    cv2.setMouseCallback('Measure', mouse_callback)
    # -------------------------------------------------------------------------

    try:
        while True:
            # 等待一对连贯的帧：深度和颜色
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 将图像转换为 numpy 数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # -------------------------- 新增部分：距离测量逻辑 --------------------------
            measure_img = color_image.copy()
            
            # 绘制测量点和连线
            if len(measure_points) >= 1:
                for pt in measure_points:
                    cv2.circle(measure_img, pt, 5, (0,0,255), -1)
                
                if len(measure_points) == 2:
                    # 获取三维坐标
                    points_3d = []
                    valid_points = True
                    for (x,y) in measure_points:
                        depth = depth_frame.get_distance(x, y)
                        if depth == 0:  # 无效深度处理
                            valid_points = False
                            break
                        # 将像素坐标转换为三维坐标
                        point = rs.rs2_deproject_pixel_to_point(
                            depth_intrinsics, [x, y], depth
                        )
                        points_3d.append(point)
                    
                    if valid_points:
                        # 计算欧氏距离
                        dx = points_3d[1][0] - points_3d[0][0]
                        dy = points_3d[1][1] - points_3d[0][1]
                        dz = points_3d[1][2] - points_3d[0][2]
                        distance = math.sqrt(dx**2 + dy**2 + dz**2)
                        distance_mm = distance * 1000  # 转换为毫米

                        # 绘制测量结果
                        cv2.line(measure_img, 
                                measure_points[0], measure_points[1],
                                (255,0,0), 2)
                        mid_point = (
                            (measure_points[0][0]+measure_points[1][0])//2,
                            (measure_points[0][1]+measure_points[1][1])//2
                        )
                        cv2.putText(measure_img, f"{distance_mm:.1f}mm",mid_point, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255), 2)
            # -------------------------------------------------------------------------

            # 对颜色图像使用 YOLOv8 进行目标检测
            results = model(color_image)
            annotated_frame = results[0].plot()

            boxes = results[0].boxes.xyxy.cpu().tolist()

            # 显示检测到的对象和估计的距离
            dectshow(color_image, boxes, depth_image)

            # 显示测量窗口
            cv2.imshow('Measure', measure_img)

            # 在深度图像上应用颜色映射
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 水平堆叠彩色和深度图像以进行显示
            images = np.hstack((color_image, depth_colormap))

            # 显示组合图像
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            # 等待按键，按 'q' 或 'ESC' 键退出
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                pipeline.stop()
                sys.exit(0)
    except Exception as e:
        print(f"发生异常: {e}")           
    finally:
        try:
            # 停止流并关闭 RealSense 管道
            pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()
