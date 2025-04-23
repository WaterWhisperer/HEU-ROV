import cv2
import numpy as np

print("NumPy 版本:", np.__version__)  # 应输出 1.26.4
print("OpenCV 版本:", cv2.__version__)  # 应输出 4.9.0
print("OpenCV imshow 功能正常:", hasattr(cv2, "imshow"))  # 应输出 True
