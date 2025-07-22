import struct
import serial
from my_packages import get_YOLO_result as gYr
import pyrealsense2 as rs
import numpy as np
import cv2

def pack_frame(data_list):
    """
    将浮点数列表打包到一个完整的串行帧中。
    Frame format:
        [0xAA, 0x55]  - Frame Header (2B)
        length        - Data byte count (1B, excludes header, footer, and checksum)
        float_data    - Little-endian float array
        [0x55, 0xAA]  - Frame Footer (2B)
    """
    if not data_list:
        return b''
    # 将浮点数组打包为字节
    data = struct.pack(f'<{len(data_list)}f', *data_list)

    if len(data) > 255:
        raise ValueError('Cannot send more than 63 floats at a time.')

    # 构建完整帧
    frame = b'\xAA\x55' + bytes([len(data)]) + data + b'\x55\xAA'
    return frame

def send_info(ser, data_list):
    """通过串行端口发送浮点数列表。"""
    if not data_list:
        print("No data to send.")
        return
    frame = pack_frame(data_list)
    ser.write(frame)
    print(f'TX ({len(frame)}B):', ' '.join(f'{b:02X}' for b in frame))

def main():
    """运行对象检测和串行通信的主要功能。"""
    # --- 配置 ---
    COLOR_FPS = 30
    DEPTH_FPS = 30
    MODEL_PATH = 'vision_marine_grasping/models/weights/test_marine_yolo11n.pt'
    SERIAL_PORT = 'COM3'
    BAUD_RATE = 115200
    CONFIDENCE_THRESHOLD = 0.75

    realsense_pipeline = None
    ser = None
    try:
        # --- 初始化 ---
        # 初始化相机与YOLO模型
        realsense_align, realsense_pipeline, realsense_profile = gYr.open_realsense_camera(COLOR_FPS, DEPTH_FPS)
        model = gYr.YOLO_init(MODEL_PATH)

        # 获取相机内参
        color_profile = realsense_profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        # 初始化串口
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Serial port {SERIAL_PORT} opened successfully.")

        # --- 主循环 ---
        while True:
            frames = realsense_pipeline.wait_for_frames()

            # 将深度与颜色框架对齐
            aligned_frames = realsense_align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            
            # YOLO 检测
            results = model(color_image)
            
            targets = gYr.get_target(results, confience_threshold=CONFIDENCE_THRESHOLD, depth_frame=depth_frame, intrinsics=color_intrinsics)
          
            # 显示检测结果
            gYr.show_debug_image(color_image, targets)

            # 退出条件
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break
                    
            # 发送目标坐标
            if targets:
                closest_target_coords = targets[0][0]
                send_info(ser, closest_target_coords)

    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # --- 退出 ---
        if realsense_pipeline:
            realsense_pipeline.stop()
            print("RealSense pipeline stopped.")
        if ser and ser.is_open:
            ser.close()
            print(f"Serial port {SERIAL_PORT} closed.")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")

if __name__ == '__main__':
    main()