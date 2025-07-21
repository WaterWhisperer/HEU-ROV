import struct
import serial
from my_packages import get_YOLO_result as gYr

def pack_frame(float_list: list[float]) -> bytes:
    """
    把一组 float 打包成完整串口帧
    帧格式：
        [0xAA, 0x55]  帧头(2B)
        length        数据字节数(1B，不含校验/帧头帧尾)
        float_data    小端 float 数组
        [0x55, 0xAA]  帧尾(2B)
    """
    #浮点数组 -> bytes，小端 4 字节
    data = struct.pack(f'<{len(float_list)}f', *float_list)

    #长度字段：数据字节数（最大 255）
    if len(data) > 255:
        raise ValueError('一次最多发送 63 个 float')

    #拼接完整帧
    frame = b'\xAA\x55' + bytes([len(data)]) + data  + b'\x55\xAA'
    return frame


def send_floats(ser: serial.Serial, float_list: list[float]) -> None:
    frame = pack_frame(float_list)
    ser.write(frame)
    print(f'TX ({len(frame)}B):', ' '.join(f'{b:02X}' for b in frame))


# -------------------------------------
if __name__ == '__main__':
    #初始化相机与YOLO模型
    color_fps = 30
    depth_fps = 30
    model_path = 'vision_marine_grasping/models/weights/test_marine_yolo11n.pt'
    realsense_align, realsense_pipeline, realsense_profile = gYr.open_realsense_camera(color_fps, depth_fps)
    model = gYr.YOLO_init(model_path)

    # 获取对齐后的彩色相机内参
    color_profile = realsense_profile.get_stream(rs.stream.color)
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    while True:
        frames = realsene_pipeline.wait_for_frames()

        # 执行深度到彩色图的对齐
        aligned_frames = realsense_align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        
        # YOLO检测
        results = model(color_image)
        boxes = results[0].boxes.xyxy.cpu().tolist()

        # 显示优化后的检测结果
        dectshow(color_image, boxes, depth_frame, color_intrinsics)

        # 退出控制
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break
                


    floats = [1.23, -4.56, 7.89e3, 0.0, 3.1415926]
    with serial.Serial('COM3', 115200, timeout=1) as ser:
        send_floats(ser, floats)