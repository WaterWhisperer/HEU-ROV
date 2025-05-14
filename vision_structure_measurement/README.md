# 三维结构物测量

该项目用于实现对三维结构物的长宽高测量

## 硬件

双目相机：Intel Realsense D455

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

Click 2 points to measure distance.
After selecting 2 points:
    Press 'l' for Length
    Press 'w' for Width
    Press 'h' for Height
Press 'c' to clear points.
Press 'x' to clear all dimensions.
Press 'p' to plot 3D (need L,W,H).
Press 'q' to quit.
