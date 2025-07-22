# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import math
# import torch
# import sys
# from ultralytics import YOLO

# def YOLO_init(model):
#     model = YOLO('vision_marine_grasping/models/weights/marine_yolo11s.pt') # 训练相关模型后，将模型路径填入''中 
#     return model

# def open_realsense_camera(color_fps, depth_fps) :
#     # 对齐对象和彩色相机内参
#     align_to = rs.stream.color
#     align = rs.align(align_to)
#     color_intrinsics = None  # 将使用对齐后的彩色相机内参

#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, color_fps)
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, depth_fps)
#     profile = pipeline.start(config)

#     return align, pipeline, profile

# def get_3d_coordinates(box, depth_frame, intrinsics, sample_size=25, window_size=5):
#     """
#     获取检测框中心区域的三维坐标（毫米）
#     Args:
#         box: 检测框坐标 [x1,y1,x2,y2]
#         depth_frame: 深度图
#         intrinsics: 相机内参
#         sample_size: 采样点数量
#         window_size: 采样窗口大小（边长的一半）
#     """
#     mid_x = int((box[0] + box[2]) // 2)
#     mid_y = int((box[1] + box[3]) // 2)
    
#     # 坐标边界检查
#     if (mid_x < window_size or mid_x >= intrinsics.width - window_size or 
#         mid_y < window_size or mid_y >= intrinsics.height - window_size):
#         return None
    
#     valid_depths = []
#     # 在中心区域随机采样
#     np.random.seed(42)  # 设置随机种子以保证结果可重复
#     for _ in range(sample_size):
#         # 在窗口范围内随机选择采样点
#         sample_x = mid_x + np.random.randint(-window_size, window_size + 1)
#         sample_y = mid_y + np.random.randint(-window_size, window_size + 1)
        
#         depth = depth_frame.get_distance(sample_x, sample_y)
#         if 0.1 < depth <= 6.0:  # 过滤无效值和远距离噪声
#             valid_depths.append(depth)
    
#     # 如果有效深度值太少，返回None
#     if len(valid_depths) < sample_size * 0.6:  # 至少60%的采样点有效
#         return None
    
#     # 计算平均深度
#     avg_depth = np.mean(valid_depths)
    
#     # 使用平均深度进行反投影
#     point_3d = rs.rs2_deproject_pixel_to_point(
#         intrinsics, [mid_x, mid_y], avg_depth
#     )
#     return [round(coord*1000, 1) for coord in point_3d]  # 米转毫米

# def get_target(results, confience_threshold, depth_frame, intrinsics) :
#     """获取目标的三维坐标,并按置信度排序，放回置信度高于阈值且距离最近的目标坐标
#     Args: 
#         result: YOLO检测结果
#         depth_frame: 深度图
#         intrinsics: 相机内参
#     """
#     if results is None or results.boxes is None or len(results.boxes) == 0:
#         return None
    
#     boxes = results.boxes.xyxy.cpu().numpy() #[N, 4]
#     confs = results.boxes.conf.cpu().numpy() #[N]
#     targets = []

#     for i, box in enumerate(boxes) :
#         if confs[i] < confience_threshold:
#             continue
        
#         # 获取三维坐标
#         coord_3d = get_3d_coordinates(box, depth_frame, intrinsics)
#         if coord_3d is not None:
#             targets.append((coord_3d, box, confs[i]))

#     if len(targets) == 0:
#         return None
    
#     targets.sort(key=lambda x: x[0][2]) # 按z坐标排序
#     return targets
       

# def show_debug_image(ori_image, targets):
#     img = ori_image.copy()
#     is_target = True

#     for target in targets:
#         if is_target:
#             mid_x = int((target[1][0] + target[1][2]) // 2)
#             mid_y = int((target[1][1] + target[1][3]) // 2)
#             cv2.circle(img, (mid_x, mid_y), 5, (255, 0, 0), -1)
#             is_target = False

#         cv2.rectangle(img, (int(target[1][0]), int(target[1][1])),
#                       (int(target[1][2]),int(target[1][3])), (0,0,255),2)
#         text = [f'X:{target[0][1]:.2f}mm', f'Y:{target[0][1]:.2f}mm', f'Z:{target[0][2]:.2f}mm', 
#                 f'Conf:{target[2]:.2f}']

#         # 计算显示位置
#         text_x = int((target[1][0] + target[1][2])//2)
#         text_y = int(target[1][1] - 10)  # 在框上方显示
#         if text_y < 30:  # 如果接近顶部，显示在框下
#             text_y = int(target[1][3] + 20)

#         # 逐行绘制坐标
#         font_scale = 0.5
#         line_height = 20
#         for i, line in enumerate(text):
#             y_pos = text_y + i*line_height
#             cv2.putText(img, line, 
#                     (text_x - 50, y_pos),  # 水平居中
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     font_scale, (0,255,255), 1, 
#                     lineType=cv2.LINE_AA)

#     cv2.imshow('Detection', img)
  
    

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def YOLO_init(model_path):
    """初始化YOLO模型."""
    model = YOLO(model_path) 
    return model

def open_realsense_camera(color_fps, depth_fps):
    """初始化并打开RealSense相机."""
    align = rs.align(rs.stream.color)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, depth_fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, color_fps)
    profile = pipeline.start(config)
    return align, pipeline, profile

def get_3d_coordinates(box, depth_frame, intrinsics, window_size=5):
    """
    计算目标的三维坐标.
        box: 目标的边界框 [x1, y1, x2, y2].
        depth_frame: 相机的深度图.
        intrinsics: 相机的内参.
        window_size: 采样窗口的半径.

    Returns:
        3D坐标 [X, Y, Z]，单位为毫米，如果无法计算则返回 None.
    """
    mid_x = int((box[0] + box[2]) / 2)
    mid_y = int((box[1] + box[3]) / 2)

    # 边界检查
    if not (window_size <= mid_x < intrinsics.width - window_size and
            window_size <= mid_y < intrinsics.height - window_size):
        return None
    # 从中心区域随机采样
    depths = []
    for y in range(mid_y - window_size, mid_y + window_size + 1):
        for x in range(mid_x - window_size, mid_x + window_size + 1):
            depth = depth_frame.get_distance(x, y)
            if 0.08 < depth <= 6.0:  # 检查深度值是否有效
                depths.append(depth)

    if not depths:
        return None

    avg_depth = np.mean(depths)
    # 反投影
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [mid_x, mid_y], avg_depth)
    
    return [round(coord * 1000, 1) for coord in point_3d]  # 毫米

def get_target(results, confience_threshold, depth_frame, intrinsics):
    """
    按置信度筛选检测到的目标，并返回按距离排序的目标。
    """
    if not results or not hasattr(results[0], 'boxes') or len(results[0].boxes) == 0:
        return []

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    
    targets = []
    for i, box in enumerate(boxes):
        if confs[i] >= confience_threshold:
            coord_3d = get_3d_coordinates(box, depth_frame, intrinsics)
            if coord_3d:
                targets.append((coord_3d, box, confs[i]))

    if not targets:
        return []

    # 按照Z坐标（距离）排序
    targets.sort(key=lambda x: x[0][2])
    return targets

def show_debug_image(ori_image, targets):
    """可视化检测结果和三维坐标."""
    img = ori_image.copy()
    
    if not targets:
        cv2.imshow('Detection', img)
        return

    for i, target in enumerate(targets):
        coord_3d, box, conf = target
        box_int = [int(p) for p in box]

        if i == 0:  # 可视化第一个目标的中心点
            mid_x = int((box_int[0] + box_int[2]) / 2)
            mid_y = int((box_int[1] + box_int[3]) / 2)
            cv2.circle(img, (mid_x, mid_y), 5, (255, 0, 0), -1)

        cv2.rectangle(img, (box_int[0], box_int[1]), (box_int[2], box_int[3]), (0, 0, 255), 2)
        
        text = [
            f'X: {coord_3d[0]:.2f}mm',
            f'Y: {coord_3d[1]:.2f}mm',
            f'Z: {coord_3d[2]:.2f}mm', 
            f'Conf: {conf:.2f}'
        ]

        text_x = box_int[0]
        text_y = box_int[1] - 10
        
        for j, line in enumerate(text):
            cv2.putText(img, line, (text_x, text_y - j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Detection', img)