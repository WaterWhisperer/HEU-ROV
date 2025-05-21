# 裂缝检测：宽度与深度测量

该项目用于实现通过双目相机对裂缝进行宽度与深度的测量

## 硬件

双目相机：Intel Realsense D405

## 项目结构

```bash
vision_crack_detection
├── cpp # C++代码
│   ├── CMakeLists.txt # 编译文件
│   ├── Combine-DistanceAndDepth.cpp
│   ├── Combine-DistanceAndDepthDiff.cpp
│   └── Mouse-DistanceAndDepth.cpp
├── python # Python代码
│   ├── BagFileObjectDetection.py
│   ├── Combine-DistanceAndDepth.py
│   ├── Mouse-DistanceAndDepth.py
│   ├── requirements.txt # 依赖文件
│   ├── YOLO-DistanceAndDepth-semi-auto.py
│   └── YOLO-WidthAndDepth-auto.py
├── README.md # 说明文档
├── tests # 测试代码
│   ├── test_env.py
│   └── test_realsense.py
└── weights # 权重文件
    ├── yolo11s_seg_crack.pt
    ├── yolov8m.pt
    ├── yolov8n.pt
    └── yolov8s_seg_crack.pt
```

## 环境

### 测试环境

- Ubuntu 22.04/20.04
- Python 3.8/3.11
