# 三维结构物测量

该项目用于实现对三维结构物的长宽高测量

## 硬件

双目相机：Intel Realsense D455

## 项目结构

```plaintext
vision_structure_measurement
├── README.md # 说明文档
└── src # 源代码
    ├── cpp # C++ 源代码
    │   ├── 3D-Structure-Measure.cpp
    │   └── CMakeLists.txt
    └── python # Python 源代码
        ├── 3D-Structure-Measure.py
        └── requirements.txt
```

## 使用说明

### 环境配置

```python
pip install -r ./vision_structure_measurement/requirements.txt
```

### 运行程序

```python
python vision_structure_measurement/3D-Structure-Measure.py 
```

### 使用方法

```plaintext
Click 2 points to measure distance.
After selecting 2 points:
    Press 'l' for Length
    Press 'w' for Width
    Press 'h' for Height
Press 'c' to clear points.
Press 'x' to clear all dimensions.
Press 'p' to plot 3D (need L,W,H).
Press 'q' to quit.
```
