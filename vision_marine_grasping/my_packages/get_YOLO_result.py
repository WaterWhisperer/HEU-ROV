import pyrealsense2 as rs
import numpy as np
import cv2
import math
import torch
import sys
from ultralytics import YOLO

def YOLO_init(model):
    model = YOLO('vision_marine_grasping/models/weights/marine_yolo11s.pt') # 训练相关模型后，将模型路径填入''中 
    return model

def open_realsense_camera(color_fps, depth_fps) :
    # 对齐对象和彩色相机内参
    align_to = rs.stream.color
    align = rs.align(align_to)
    color_intrinsics = None  # 将使用对齐后的彩色相机内参

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, color_fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, depth_fps)
    profile = pipeline.start(config)

    return align, pipeline, profile

def get_3d_coordinates(box, depth_frame, intrinsics, sample_size=25, window_size=5):
    """
    获取检测框中心区域的三维坐标（毫米）
    Args:
        box: 检测框坐标 [x1,y1,x2,y2]
        depth_frame: 深度图
        intrinsics: 相机内参
        sample_size: 采样点数量
        window_size: 采样窗口大小（边长的一半）
    """
    mid_x = int((box[0] + box[2]) // 2)
    mid_y = int((box[1] + box[3]) // 2)
    
    # 坐标边界检查
    if (mid_x < window_size or mid_x >= intrinsics.width - window_size or 
        mid_y < window_size or mid_y >= intrinsics.height - window_size):
        return None
    
    valid_depths = []
    # 在中心区域随机采样
    np.random.seed(42)  # 设置随机种子以保证结果可重复
    for _ in range(sample_size):
        # 在窗口范围内随机选择采样点
        sample_x = mid_x + np.random.randint(-window_size, window_size + 1)
        sample_y = mid_y + np.random.randint(-window_size, window_size + 1)
        
        depth = depth_frame.get_distance(sample_x, sample_y)
        if 0.1 < depth <= 6.0:  # 过滤无效值和远距离噪声
            valid_depths.append(depth)
    
    # 如果有效深度值太少，返回None
    if len(valid_depths) < sample_size * 0.6:  # 至少60%的采样点有效
        return None
    
    # 计算平均深度
    avg_depth = np.mean(valid_depths)
    
    # 使用平均深度进行反投影
    point_3d = rs.rs2_deproject_pixel_to_point(
        intrinsics, [mid_x, mid_y], avg_depth
    )
    return [round(coord*1000, 1) for coord in point_3d]  # 米转毫米

def get_target(result, depth_frame, intrinsics) :

    pass




