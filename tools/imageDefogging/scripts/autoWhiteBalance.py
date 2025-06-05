'''
文件名：autoWhiteBalance.py
功能：实现自动白平衡算法
作者: WaterWhisperer
创建日期: 2024-12-04
更新日期: 2025-06-05
版权: 自由使用，保留作者信息
'''

import cv2
import numpy as np

def motowhitebalance(originalImage):
    """
    实现自动白平衡算法
    Args:
        originalImage: 输入的BGR格式图像
    Returns:
        adjustedImage: 白平衡调整后的图像
    """
    # 将图像从RGB颜色空间转换到YCbCr颜色空间
    # YCbCr更适合处理色彩信息
    im1 = cv2.cvtColor(originalImage, cv2.COLOR_RGB2YCrCb)
    Lu, Cb, Cr = cv2.split(im1)  # Lu为亮度分量，Cb和Cr为色度分量
    
    # 获取图像尺寸用于分块处理
    x, y, _ = im1.shape
    
    def test(Cb, Cr, x, y):
        """
        计算色度分量的统计特征
        Args:
            Cb, Cr: 色度分量
            x, y: 块的尺寸
        Returns:
            Mb, Mr: 色度分量的平均值
            Db, Dr: 色度分量的偏差
        """
        Mb = np.mean(Cb)
        Mr = np.mean(Cr)
        Db = np.sum(Cb - Mb) / (x * y)
        Dr = np.sum(Cr - Mr) / (x * y)
        return Mb, Mr, Db, Dr
    
    # 将图像分为四个块进行处理，提高局部自适应性
    # 第一块（左上）
    I1 = im1[:x//2, :y//2, :]
    Mb1, Mr1, Db1, Dr1 = test(Cb[:x//2, :y//2], Cr[:x//2, :y//2], x//2, y//2)
    
    # 第二块（左下）
    I2 = im1[x//2:, :y//2, :]
    _, _, _, _ = test(Cb[x//2:, :y//2], Cr[x//2:, :y//2], x//2, y//2)
    # Mb2, Mr2, Db2, Dr2 = test(Cb[:x//2, :y//2], Cr[:x//2, :y//2], x//2, y//2)

    # 第三块（右上）
    I3 = im1[:x//2, y//2:, :]
    Mb3, Mr3, Db3, Dr3 = test(Cb[:x//2, y//2:], Cr[:x//2, y//2:], x//2, y//2)
    
    # 第四块（右下）
    I4 = im1[x//2:, y//2:, :]
    Mb4, Mr4, Db4, Dr4 = test(Cb[x//2:, y//2:], Cr[x//2:, y//2:], x//2, y//2)
    
    # 计算整体统计特征（忽略第二块，水下图像左下区域通常容易受影响）
    # 经验表明，左下区域可能包含更多阴影或噪声，因此忽略该区域的统计特征
    Mr = (Mr1 + Mr3 + Mr4) / 3
    Mb = (Mb1 + Mb3 + Mb4) / 3
    Dr = (Dr1 + Dr3 + Dr4) / 3
    Db = (Db1 + Db3 + Db4) / 3
    
    # 如果需要考虑第二块，包含所有四个块
    # Mr = (Mr1 + Mr2 + Mr3 + Mr4) / 4
    # Mb = (Mb1 + Mb2 + Mb3 + Mb4) / 4
    # Dr = (Dr1 + Dr2 + Dr3 + Dr4) / 4
    # Db = (Db1 + Db2 + Db3 + Db4) / 4

    # 提取符合条件的亮度值
    Ciny = []
    tst = np.zeros_like(Lu)
    for i in range(x):
        for j in range(y):
            # 检查色度值是否在阈值范围内
            b1 = Cb[i,j] - (Mb + Db * np.sign(Mb))
            b2 = Cr[i,j] - (1.5 * Mr + Dr * np.sign(Mr))
            if (b1 < abs(1.5 * Db) and b2 < abs(1.5 * Dr)):
                Ciny.append(Lu[i,j])
                tst[i,j] = Lu[i,j]
    
    # 对亮度值进行排序和选择
    Ciny = np.array(Ciny)
    sumsort = np.sort(Ciny)[::-1]  # 降序排序
    count = round(len(sumsort) / 10)  # 选取前10%的亮度值
    Ciny2 = sumsort[:count]
    mn = np.min(Ciny2)
    index = Lu > mn  # 创建亮度蒙版
    
    # 计算每个通道的增益
    R = originalImage[:,:,0]
    G = originalImage[:,:,1]
    B = originalImage[:,:,2]
    Rave = np.mean(R[index])
    Gave = np.mean(G[index])
    Bave = np.mean(B[index])
    Ymax = np.max(Lu)
    
    # 计算增益系数
    Rgain = Ymax / Rave
    Ggain = Ymax / Gave
    Bgain = Ymax / Bave
    
    # 应用增益调整
    R = R * Rgain
    G = G * Ggain
    B = B * Bgain
    adjustedImage = cv2.merge((R, G, B))
    
    return adjustedImage

# 主程序
if __name__ == "__main__":
    # 读取图像
    originalImage = cv2.imread('/home/water/桌面/ROV/data1.jpg')
    if originalImage is None:
        print('Could not read the image.')
        exit(1)
    
    # 应用白平衡调整
    adjustedImage = motowhitebalance(originalImage)
    
    # 显示结果
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image', originalImage)
    cv2.namedWindow('Adjusted Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Adjusted Image', adjustedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()