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