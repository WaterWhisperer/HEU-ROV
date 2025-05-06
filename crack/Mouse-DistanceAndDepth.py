import pyrealsense2 as rs
import numpy as np
import cv2
import random
import math

# 全局变量
box_points = []       # 框选坐标 [start, end]
meas_points = []      # 测量点坐标 [point1, point2]
depth_intrinsics = None
current_window = ""   # 当前活动窗口

# 专用鼠标回调函数
def create_mouse_handler(window_name):
    def handler(event, x, y, flags, param):
        global box_points, meas_points, current_window
        
        current_window = window_name
        
        # 框选窗口处理（左键拖拽）
        if window_name == "Depth Measurement":
            # 左键按下：记录起点
            if event == cv2.EVENT_LBUTTONDOWN:
                box_points = [(x, y)]  
            # 左键拖拽：实时更新终点
            elif event == cv2.EVENT_MOUSEMOVE:
                if flags & cv2.EVENT_FLAG_LBUTTON:  # 修正判断条件
                    if len(box_points) == 1:
                        box_points.append((x, y))
                    elif len(box_points) == 2:
                        box_points[1] = (x, y)  # 持续更新终点坐标           
            # 左键释放：保持最终坐标
            elif event == cv2.EVENT_LBUTTONUP:
                if len(box_points) == 2:
                    box_points[1] = (x, y)       

        # 点测窗口处理（左键点击）
        elif window_name == "Distance Measurement":
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(meas_points) < 2:
                    meas_points.append((x, y))
                else:
                    meas_points = [(x, y)]
    return handler


# 框选深度计算
def calculate_box_depth(box, depth_data):
    if len(box) != 2:
        return 0
    
    # 规范坐标
    x1, y1 = min(box[0][0], box[1][0]), min(box[0][1], box[1][1])
    x2, y2 = max(box[0][0], box[1][0]), max(box[0][1], box[1][1])
    
    # 中心区域采样
    samples = []
    for _ in range(50):
        px = random.randint(x1, x2)
        py = random.randint(y1, y2)
        if 0 <= px < depth_data.shape[1] and 0 <= py < depth_data.shape[0]:
            dist = depth_data[py, px] / 10  # 转毫米
            if dist > 0:
                samples.append(dist)
                
    return np.median(samples) if samples else 0

# 三维距离计算
def calculate_3d_distance(points):
    if len(points) != 2:
        return 0
    
    try:
        point3d = []
        for (x, y) in points:
            depth = depth_frame.get_distance(x, y)
            point3d.append(rs.rs2_deproject_pixel_to_point(
                depth_intrinsics, [x, y], depth))
            
        dx = point3d[1][0] - point3d[0][0]
        dy = point3d[1][1] - point3d[0][1]
        dz = point3d[1][2] - point3d[0][2]
        return math.sqrt(dx**2 + dy**2 + dz**2) * 1000  # 毫米
    except:
        return 0

if __name__ == "__main__":
    # 配置相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    profile = pipeline.start(config)
    
    # 获取相机参数
    depth_profile = profile.get_stream(rs.stream.depth)
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
    
    # 创建双窗口
    cv2.namedWindow("Depth Measurement")
    cv2.namedWindow("Distance Measurement")
    cv2.setMouseCallback("Depth Measurement", create_mouse_handler("Depth Measurement")) 
    cv2.setMouseCallback("Distance Measurement", create_mouse_handler("Distance Measurement"))
    
    try:
        while True:
            # 获取帧数据
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame: continue
            
            # 准备基础图像
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            box_img = color_image.copy()  # 框选显示图像
            point_img = color_image.copy() # 点测显示图像
            
            # 框选窗口处理
            if len(box_points) >= 1:
                # 实时绘制拖拽框
                if current_window == "Depth Measurement" and len(box_points) == 2:
                    cv2.rectangle(box_img, box_points[0], box_points[1], (0,200,0), 2)
                    
                # 完成测量后显示结果
                if len(box_points) == 2:
                    depth = calculate_box_depth(box_points, depth_image)
                    cv2.rectangle(box_img, box_points[0], box_points[1], (0,255,0), 2)
                    cv2.putText(box_img, f"Depth: {depth:.1f}mm", 
                    (box_points[0][0], box_points[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    
            # 点测窗口处理
            if len(meas_points) >= 1:
                # 绘制测量点
                for pt in meas_points:
                    cv2.circle(point_img, pt, 5, (50,50,255), -1)
                    
                # 绘制测量线及结果
                if len(meas_points) == 2:
                    cv2.line(point_img, meas_points[0], meas_points[1], (255,100,100), 2)
                    distance = calculate_3d_distance(meas_points)
                    mid_x = (meas_points[0][0] + meas_points[1][0]) // 2
                    mid_y = (meas_points[0][1] + meas_points[1][1]) // 2
                    cv2.putText(point_img, f"Distance: {distance:.1f}mm",
                    (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
            # 显示处理
            cv2.imshow("Depth Measurement", box_img)
            cv2.imshow("Distance Measurement", point_img)
            
            # 退出控制
            if cv2.waitKey(1) in [ord('q'), 27]:
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
