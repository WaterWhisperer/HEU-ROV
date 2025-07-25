# 裂缝检测：宽度与深度测量

该项目用于实现通过双目相机对裂缝进行宽度与深度的测量

## 硬件

双目相机：Intel Realsense D405

## 项目结构

```plaintext
vision_crack_detection
├── models  # 模型文件
│   ├── README.md
│   └── weights # 权重文件
│       ├── crack_yolo11s_seg.pt
│       ├── crack_yolov8m.pt
│       ├── crack_yolov8n.pt
│       └── crack_yolov8s_seg.pt
├── README.md  # 说明文档
├── src  # 代码文件
│   ├── cpp  # C++代码
│   │   ├── CMakeLists.txt
│   │   ├── Combine-DistanceAndDepth.cpp
│   │   ├── Combine-DistanceAndDepthDiff.cpp
│   │   └── Mouse-DistanceAndDepth.cpp
│   └── python  # Python代码
│       ├── BagFileObjectDetection.py
│       ├── Combine-DistanceAndDepth.py
│       ├── Mouse-DistanceAndDepth.py
│       ├── requirements.txt
│       ├── YOLO-DistanceAndDepth-semi-auto.py
│       └── YOLO-WidthAndDepth-auto.py
└── tests  # 测试文件
    ├── cpp  # C++测试文件
    └── python  # Python测试文件
        ├── test_env.py
        └── test_realsense.py
```

## 环境

### 测试环境

- Ubuntu 22.04/20.04
- Python 3.8/3.11

## 使用说明

各个代码的使用说明位于对应代码文件的开头部分
