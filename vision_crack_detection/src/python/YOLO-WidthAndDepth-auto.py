'''
文件名：YOLO-WidthAndDepth-auto.py
功能：使用YOLO模型进行目标检测，并自动输出识别框的深度和宽度信息。
'''
import pyrealsense2 as rs
import numpy as np
import cv2
import random
import math
from ultralytics import YOLO

# 初始化YOLO模型
model = YOLO('vision_crack_detection/models/weights/crack_yolov8s_seg.pt')  # 请确认模型路径正确

# RealSense配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动流并获取内参
profile = pipeline.start(config)
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()

# 创建对齐对象（深度到彩色）
align_to = rs.stream.color
align = rs.align(align_to)

def get_mid_pos(frame, box, depth_frame, randnum):
    """使用get_distance获取精确米单位深度"""
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2]
    min_val = min(abs(box[2]-box[0]), abs(box[3]-box[1]))

    for _ in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        x = int(mid_pos[0] + bias)
        y = int(mid_pos[1] + bias)
        dist = depth_frame.get_distance(x, y)  # 直接获取米单位值
        if dist > 0:
            distance_list.append(dist * 1000)  # 转换为毫米
    
    if distance_list:
        distance_list = sorted(distance_list)[len(distance_list)//4 : -len(distance_list)//4]
        return np.mean(distance_list)
    return 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # YOLO推理
        results = model(color_image, conf=0.3)
        annotated_frame = results[0].plot()

        if results[0].masks is not None:
            for mask in results[0].masks.xy:
                if len(mask) < 1:
                    continue

                # 获取边界框
                x_coords = mask[:, 0]
                y_coords = mask[:, 1]
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)

                # 计算中心深度
                depth = get_mid_pos(color_image, [x_min, y_min, x_max, y_max], depth_frame, 24)
                
                # 计算物理尺寸
                real_coords = []
                for px, py in [(x_min, y_min), (x_max, y_max)]:
                    depth_val = depth_frame.get_distance(int(px), int(py))
                    if depth_val > 0:
                        X = (px - color_intrinsics.ppx) * depth_val / color_intrinsics.fx
                        Y = (py - color_intrinsics.ppy) * depth_val / color_intrinsics.fy
                        real_coords.append((X, Y))
                
                if len(real_coords) >= 2:
                    real_width = (real_coords[1][0] - real_coords[0][0]) * 1000  # 宽度(mm)
                    
                    # 原始置信度信息（由plot自动生成在左上）
                    # 新增深度/宽度信息显示在框下方
                    text_pos_y = int(y_max) + 20  # 在框底部下方20像素
                    text_line1 = f"Depth: {depth:.1f}mm"
                    text_line2 = f"Width: {abs(real_width):.1f}mm"
                    
                    # 绘制第一行文本
                    cv2.putText(annotated_frame, text_line1,
                            (int(x_min), text_pos_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    
                    # 绘制第二行文本（向下偏移30像素）
                    cv2.putText(annotated_frame, text_line2,
                            (int(x_min), text_pos_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow('Crack Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()