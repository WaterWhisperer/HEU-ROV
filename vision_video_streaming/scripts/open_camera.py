'''
安装依赖：
pip install opencv-python

运行：
python open_camera.py
'''
import cv2

def main():
    # 打开摄像头（默认设备索引0）
    cap = cv2.VideoCapture(0)
    
    # 设置分辨率（可调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("按 'q' 键退出测试...")
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("错误：无法从摄像头获取帧")
            break
            
        # 显示画面
        cv2.imshow('Camera Preview - Press Q to Exit', frame)
        
        # 检测按键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
