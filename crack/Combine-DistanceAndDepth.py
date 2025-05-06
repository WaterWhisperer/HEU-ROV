'''
文件名：Combine-DistanceAndDepth.py
功能：深度测量和距离测量的结合，显示在同一窗口中。
    Depth Measurement：左侧视图，用于深度测量。
    Distance Measurement：右侧视图，用于距离测量。
'''
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
WINDOW_WIDTH = 640    # 单个视图宽度

# 专用鼠标回调函数
def create_mouse_handler():
    def handler(event, x, y, flags, param):
        global box_points, meas_points, current_window
        
        # 判断操作区域（左侧深度测量/右侧距离测量）
        current_window = "Depth" if x < WINDOW_WIDTH else "Distance"
        # 坐标转换到对应视图
        adj_x = x % WINDOW_WIDTH
        
        # 深度测量区域处理（左侧）
        if current_window == "Depth":
            # 左键按下：记录起点
            if event == cv2.EVENT_LBUTTONDOWN:
                box_points = [(adj_x, y)]  
            # 左键拖拽：实时更新终点
            elif event == cv2.EVENT_MOUSEMOVE:
                if flags & cv2.EVENT_FLAG_LBUTTON:
                    if len(box_points) == 1:
                        box_points.append((adj_x, y))
                    elif len(box_points) == 2:
                        box_points[1] = (adj_x, y)           
            # 左键释放：保持最终坐标
            elif event == cv2.EVENT_LBUTTONUP:
                if len(box_points) == 2:
                    box_points[1] = (adj_x, y)

        # 距离测量区域处理（右侧）
        elif current_window == "Distance":
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(meas_points) < 2:
                    meas_points.append((adj_x, y))
                else:
                    meas_points = [(adj_x, y)]
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
    
    # 创建合并窗口
    cv2.namedWindow("Measurement System")
    cv2.setMouseCallback("Measurement System", create_mouse_handler())
    
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
            left_img = color_image.copy()   # 左侧视图
            right_img = color_image.copy()  # 右侧视图
            
            # 深度测量处理（左侧）
            if len(box_points) >= 1:
                if current_window == "Depth" and len(box_points) == 2:
                    cv2.rectangle(left_img, box_points[0], box_points[1], (0,200,0), 2)
                    
                if len(box_points) == 2:
                    depth = calculate_box_depth(box_points, depth_image)
                    cv2.rectangle(left_img, box_points[0], box_points[1], (0,255,0), 2)
                    cv2.putText(left_img, f"Depth: {depth:.1f}mm", 
                    (box_points[0][0], box_points[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
            # 距离测量处理（右侧）
            if len(meas_points) >= 1:
                for pt in meas_points:
                    cv2.circle(right_img, pt, 5, (50,50,255), -1)
                    
                if len(meas_points) == 2:
                    cv2.line(right_img, meas_points[0], meas_points[1], (255,100,100), 2)
                    distance = calculate_3d_distance(meas_points)
                    mid_x = (meas_points[0][0] + meas_points[1][0]) // 2
                    mid_y = (meas_points[0][1] + meas_points[1][1]) // 2
                    cv2.putText(right_img, f"Distance: {distance:.1f}mm",
                    (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
            # 合并显示
            combined = np.hstack((left_img, right_img))
            # 添加分割线
            cv2.line(combined, (WINDOW_WIDTH,0), (WINDOW_WIDTH,480), (200,200,200), 2)
            # 添加文字标注
            cv2.putText(combined, "Depth Measurement", (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(combined, "Distance Measurement", (WINDOW_WIDTH+10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            cv2.imshow("Measurement System", combined)
            
            # 退出控制
            if cv2.waitKey(1) in [ord('q'), 27]:
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
