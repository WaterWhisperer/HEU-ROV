'''
文件名：3D-Structure-Measure.py
功能：使用双目相机测量3D结构物的长度、宽度和高度。
使用方法：
    Click 2 points to measure distance.
    After selecting 2 points:
        Press 'l' for Length
        Press 'w' for Width
        Press 'h' for Height
    Press 'c' to clear points.
    Press 'x' to clear all dimensions.
    Press 'p' to plot 3D (need L,W,H).
    Press 'm' to toggle manual mode.
    Press 'q' to quit.
'''
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Text3D
# from PIL import ImageFont, ImageDraw, Image # 如果需要中文文字渲染，注释掉此行

# --- Configuration ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
LINE_COLOR = (0, 255, 0)
POINT_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)

# --- Global Variables ---
selected_points_2d = []
current_measurement_3d_points = []
measurements_log = {'Length': None, 'Width': None, 'Height': None} # Store L, W, H
intrinsics = None
depth_scale = 1.0
typing_mode = False # Not used in this version, direct assignment
current_label_input = "" # Not used in this version
show_help = False  # Control help display

# --- Helper Functions ---
def deproject_pixel_to_point_custom(pixel, depth, intrinsics_obj):
    x = (pixel[0] - intrinsics_obj.ppx) / intrinsics_obj.fx
    y = (pixel[1] - intrinsics_obj.ppy) / intrinsics_obj.fy
    return [depth * x, depth * y, depth]

def calculate_3d_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def mouse_callback(event, x, y, flags, param):
    global selected_points_2d, current_measurement_3d_points, intrinsics, depth_frame_global

    if event == cv2.EVENT_LBUTTONDOWN:
        if depth_frame_global and intrinsics:
            if len(selected_points_2d) < 2:
                depth_value = depth_frame_global.get_distance(x, y)
                if depth_value == 0:
                    print(f"警告：选定点 ({x}, {y}) 深度为0。未添加点。请尝试其他点。")
                    return

                selected_points_2d.append((x, y))
                point_3d = deproject_pixel_to_point_custom([x, y], depth_value, intrinsics)
                current_measurement_3d_points.append(point_3d)
                print(f"已选点 {len(selected_points_2d)}: 2D=({x},{y}), 3D=({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})m")

                if len(selected_points_2d) == 2:
                    dist = calculate_3d_distance(current_measurement_3d_points[0], current_measurement_3d_points[1])
                    print(f"两点间距离: {dist:.3f} 米")
                    print("按 'l' 保存为长度, 'w' 为宽度, 'h' 为高度, 或 'c' 清除。")
            else:
                print("已选择2个点。按 'l'/'w'/'h' 保存，或按 'c' 清除。")
        else:
            print("深度帧或内参尚未准备好。")

# 处理中文文字渲染，由于使用中文渲染会导致程序运行速度变慢，因此注释掉此函数
# def cv2_add_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
#     """
#     在 opencv 图片上显示中文
#     :param img: opencv 图片
#     :param text: 要添加的文字
#     :param position: 文字位置 (x, y)
#     :param textColor: 文字颜色
#     :param textSize: 文字大小
#     """
#     if isinstance(img, np.ndarray):
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
#     # 创建一个同样大小的透明图层
#     txt = Image.new('RGBA', img.size, (0, 0, 0, 0))
    
#     # 加载一个中文字体文件
#     font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", textSize)
    
#     draw = ImageDraw.Draw(txt)
#     draw.text(position, text, font=font, fill=textColor)
    
#     # 将文字图层覆盖到原图
#     combined = Image.alpha_composite(img.convert('RGBA'), txt)
#     return cv2.cvtColor(np.asarray(combined), cv2.COLOR_RGB2BGR)

def draw_ui(image):
    global selected_points_2d, current_measurement_3d_points, measurements_log, show_help

    y_offset = 30
    # Display help only if enabled
    if show_help:
        # English instructions
        instructions = [
            "Click 2 points to measure distance.",
            "After selecting 2 points:",
            "  Press 'l' for Length",
            "  Press 'w' for Width",
            "  Press 'h' for Height",
            "Press 'c' to clear points.",
            "Press 'x' to clear all dimensions.",
            "Press 'p' to plot 3D (need L,W,H).",
            "Press 'm' to toggle manual mode.",
            "Press 'q' to quit."
        ]

        # 使用普通OpenCV文字渲染替代中文渲染
        for i, instruction in enumerate(instructions):
            cv2.putText(image, instruction, 
                       (10, y_offset + i * 25),
                        FONT, FONT_SCALE, TEXT_COLOR, 
                        FONT_THICKNESS)
        
        log_y_start = y_offset + len(instructions) * 25 + 10
    else:
        # Display minimal instructions when help is hidden
        cv2.putText(image, "Press 'm' for manual.", 
                    (10, y_offset),
                    FONT, FONT_SCALE, TEXT_COLOR, 
                    FONT_THICKNESS)
        log_y_start = y_offset + 30

    # Display points with boundary checks
    for i, p_2d in enumerate(selected_points_2d):
        cv2.circle(image, p_2d, 2, POINT_COLOR, -1)
        p3d = current_measurement_3d_points[i]
        text = f"P{i+1} ({p3d[0]:.2f}, {p3d[1]:.2f}, {p3d[2]:.2f})m"
        
        # Calculate text size for boundary check
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, 0.5, FONT_THICKNESS)
        
        # Adjust position if near right edge
        x_pos = p_2d[0] + 10
        if x_pos + text_width > image.shape[1]:
            x_pos = p_2d[0] - text_width - 10
            
        # Adjust position if near top edge
        y_pos = p_2d[1] - 10
        if y_pos - text_height < 0:
            y_pos = p_2d[1] + text_height + 10
            
        cv2.putText(image, text,
                    (x_pos, y_pos),
                    FONT, 0.5, POINT_COLOR,
                    FONT_THICKNESS)

    # Display distance with boundary checks
    if len(selected_points_2d) == 2:
        cv2.line(image, selected_points_2d[0], selected_points_2d[1], LINE_COLOR, 2)
        dist = calculate_3d_distance(current_measurement_3d_points[0], current_measurement_3d_points[1])
        mid_point = ((selected_points_2d[0][0] + selected_points_2d[1][0]) // 2,
                    (selected_points_2d[0][1] + selected_points_2d[1][1]) // 2)
        
        text = f"{dist:.3f} m"
        # Calculate text size for boundary check
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, 0.7, FONT_THICKNESS)
        
        # Adjust position if near right edge
        x_pos = mid_point[0]
        if x_pos + text_width/2 > image.shape[1]:
            x_pos = image.shape[1] - text_width - 5
        elif x_pos - text_width/2 < 0:
            x_pos = text_width/2 + 5
            
        # Adjust position if near top edge
        y_pos = mid_point[1] - 10
        if y_pos - text_height < 0:
            y_pos = mid_point[1] + text_height + 10
            
        cv2.putText(image, text,
                    (int(x_pos), int(y_pos)),
                    FONT, 0.7, LINE_COLOR,
                    FONT_THICKNESS)

    # Display saved dimensions
    cv2.putText(image, "Saved Dimensions:",
                (10, log_y_start),
                FONT, 0.7, TEXT_COLOR,
                FONT_THICKNESS)

    for i, (dim, value) in enumerate(measurements_log.items()):
        text = f"- {dim}: {value:.3f} m" if value is not None else f"- {dim}: Not Set"
        cv2.putText(image, text,
                   (10, log_y_start + 20 + i * 25),
                    FONT, 0.7, TEXT_COLOR,
                    FONT_THICKNESS)

    return image

def plot_3d_structure(length, width, height):
    if length is None or width is None or height is None:
        print("Error: Need to set Length, Width, and Height to plot 3D structure.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 定义立方体的8个顶点
    # 假设长度沿x轴，宽度沿y轴，高度沿z轴，且一个角点在原点
    vertices = np.array([
        [0, 0, 0],
        [length, 0, 0],
        [length, width, 0],
        [0, width, 0],
        [0, 0, height],
        [length, 0, height],
        [length, width, height],
        [0, width, height]
    ])

    # 定义立方体的6个面（每个面由4个顶点索引组成）
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]], # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]], # 顶面
        [vertices[0], vertices[1], vertices[5], vertices[4]], # 前面
        [vertices[2], vertices[3], vertices[7], vertices[6]], # 后面
        [vertices[1], vertices[2], vertices[6], vertices[5]], # 右面
        [vertices[0], vertices[3], vertices[7], vertices[4]]  # 左面
    ]

    # 创建Poly3DCollection对象
    ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    # 标注尺寸
    # 长度
    ax.text(length/2, -width*0.1, 0, 
            f"L: {length:.3f}m", 
            color='black', 
            ha='center')
    ax.plot([0, length], [-width*0.05, -width*0.05], [0,0], 'k-|')
    # 宽度 - 修复rotation参数
    ax.text(-length*0.1, width/2, 0, 
            f"W: {width:.3f}m", 
            color='black', 
            ha='center', 
            rotation=90,  # 改为90度而不是'z'
            va='center')
    ax.plot([-length*0.05, -length*0.05], [0, width], [0,0], 'k-|')
    # 高度
    ax.text(-length*0.1, -width*0.1, height/2, 
            f"H: {height:.3f}m", 
            color='black', 
            ha='center')
    ax.plot([-length*0.05, -width*0.05], [-width*0.05, -width*0.05], [0, height], 'k-|')


    ax.set_xlabel('Length (X)')
    ax.set_ylabel('Width (Y)')
    ax.set_zlabel('Height (Z)')
    ax.set_title('Measured 3D Structure')

    # 设置坐标轴范围以更好地显示立方体
    ax.set_xlim([-length*0.2, length * 1.2])
    ax.set_ylim([-width*0.2, width * 1.2])
    ax.set_zlim([-height*0.2, height * 1.2])
    ax.view_init(elev=20., azim=-35) #调整视角
    plt.tight_layout()
    plt.show()

# --- Main Function ---
depth_frame_global = None

def main():
    global selected_points_2d, current_measurement_3d_points, measurements_log, intrinsics, depth_frame_global, depth_scale, show_help

    pipeline = rs.pipeline()
    config = rs.config()

    # 尝试启用更高精度的预设或设置
    # 注意：这些设置可能需要根据您的具体librealsense版本和相机固件进行调整
    # config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    # 查找支持的流配置
    try:
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()

        # 尝试设置高精度预设
        if depth_sensor.supports(rs.option.visual_preset):
            # RS2_VISUAL_PRESET_HIGH_ACCURACY = 1
            # RS2_VISUAL_PRESET_HIGH_DENSITY = 2
            # RS2_VISUAL_PRESET_MEDIUM_DENSITY = 3
            # RS2_VISUAL_PRESET_HAND = 4
            # RS2_VISUAL_PRESET_LOW_AMBIENT = 5
            # RS2_VISUAL_PRESET_MAX_RANGE = 6
            # RS2_VISUAL_PRESET_CUSTOM = 0
            # print("尝试设置高精度预设...")
            # depth_sensor.set_option(rs.option.visual_preset, rs.rs2_visual_preset.RS2_VISUAL_PRESET_HIGH_ACCURACY) # Python中枚举访问方式不同
            preset_val = 1 # High Accuracy
            try:
                depth_sensor.set_option(rs.option.visual_preset, preset_val)
                print(f"已设置视觉预设为: {preset_val} (高精度)")
            except Exception as e:
                print(f"无法设置高精度预设: {e}")
        else:
            print("设备不支持视觉预设选项。")

        # 配置流 (选择一个可用的分辨率和帧率)
        # 您可以使用 rs-enumerate-devices 工具查看支持的模式
        depth_stream_profile = None
        color_stream_profile = None

        for s in depth_sensor.get_stream_profiles():
            if s.stream_type() == rs.stream.depth and s.format() == rs.format.z16 and s.fps() == 30:
                if s.as_video_stream_profile().width() == 848 and s.as_video_stream_profile().height() == 480: # 尝试这个分辨率
                    depth_stream_profile = s
                    break
        if not depth_stream_profile: # 如果找不到848x480，尝试1280x720
            for s in depth_sensor.get_stream_profiles():
                if s.stream_type() == rs.stream.depth and s.format() == rs.format.z16 and s.fps() == 30:
                    if s.as_video_stream_profile().width() == 1280 and s.as_video_stream_profile().height() == 720:
                        depth_stream_profile = s
                        break
        if not depth_stream_profile:
            print("错误: 找不到合适的深度流配置。请检查相机连接和支持的模式。")
            return

        rgb_sensor = device.first_color_sensor()
        if not rgb_sensor:
            print("错误: 找不到颜色传感器。")
            return

        for s in rgb_sensor.get_stream_profiles():
            if s.stream_type() == rs.stream.color and s.format() == rs.format.bgr8 and s.fps() == 30:
                if s.as_video_stream_profile().width() == depth_stream_profile.as_video_stream_profile().width() and \
                    s.as_video_stream_profile().height() == depth_stream_profile.as_video_stream_profile().height():
                    color_stream_profile = s
                    break
        if not color_stream_profile: # 如果找不到匹配的分辨率，尝试一个常见的
            for s in rgb_sensor.get_stream_profiles():
                if s.stream_type() == rs.stream.color and s.format() == rs.format.bgr8 and s.fps() == 30:
                    if s.as_video_stream_profile().width() == 848 and s.as_video_stream_profile().height() == 480:
                        color_stream_profile = s
                        break
        if not color_stream_profile:
            print("错误: 找不到合适的颜色流配置。")
            return

        print(f"使用深度流: {depth_stream_profile}")
        print(f"使用颜色流: {color_stream_profile}")
        width = depth_stream_profile.as_video_stream_profile().width()
        height = depth_stream_profile.as_video_stream_profile().height()
        fps = depth_stream_profile.fps()

        # 正确配置深度和彩色流
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    except Exception as e:
        print(f"配置 RealSense 时出错: {e}")
        return

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例尺: {depth_scale}")

    align_to = rs.stream.color
    align = rs.align(align_to)

    # 获取对齐后的颜色流的内参
    color_profile_aligned = profile.get_stream(rs.stream.color)
    intrinsics = color_profile_aligned.as_video_stream_profile().get_intrinsics()
    print(f"颜色流内参 (对齐后): {intrinsics}")

    cv2.namedWindow('ROV Measurement - D455', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('ROV Measurement - D455', mouse_callback)

    print("相机已初始化。在图像上点击两个点以测量距离。")
    print("按键说明见图像左上角。")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_frame_global = aligned_depth_frame
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03 / depth_scale), cv2.COLORMAP_JET)

            display_image = color_image.copy()
            draw_ui(display_image)
            images = np.hstack((display_image, depth_colormap))
            cv2.imshow('ROV Measurement - D455', images)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                selected_points_2d.clear()
                current_measurement_3d_points.clear()
                print("当前点已清除。")
            elif key == ord('x'):
                measurements_log = {'Length': None, 'Width': None, 'Height': None}
                print("所有已存尺寸已清除。")
            elif key in [ord('l'), ord('w'), ord('h')]:
                if len(current_measurement_3d_points) == 2:
                    dist = calculate_3d_distance(current_measurement_3d_points[0], current_measurement_3d_points[1])
                    dimension_type = ""
                    if key == ord('l'): dimension_type = 'Length'
                    elif key == ord('w'): dimension_type = 'Width'
                    elif key == ord('h'): dimension_type = 'Height'

                    measurements_log[dimension_type] = dist
                    print(f"已保存 {dimension_type}: {dist:.3f} m")
                    selected_points_2d.clear()
                    current_measurement_3d_points.clear()
                else:
                    print("请先选择2个点来定义一个测量。")
            elif key == ord('m'):
                show_help = not show_help
            elif key == ord('p'):
                plot_3d_structure(measurements_log['Length'], measurements_log['Width'], measurements_log['Height'])

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("流程已停止，窗口已关闭。")

if __name__ == "__main__":
    main()
