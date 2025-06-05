/*
 * 文件名：imageHalf.cpp
 * 功能：将图像分割为左半部分图像并保存
 */ 

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像
    std::string imagePath = "/home/water/桌面/opencv/xxx.jpg";
    cv::Mat image = cv::imread(imagePath);

    // 检查图像是否成功加载
    if (image.empty()) {
        std::cout << "未能加载图像，请检查路径。" << std::endl;
        return -1;
    }

    // 获取图像的高度和宽度
    int width = image.cols;
    int height = image.rows;

    // 计算左半部分的边界
    cv::Rect leftHalf(0, 0, width / 2, height);
    cv::Mat leftImage = image(leftHalf);

    // 保存左半部分图像
    std::string leftHalfPath = "/home/water/桌面/opencv/xxx.jpg";
    cv::imwrite(leftHalfPath, leftImage);

    std::cout << "左半部分图像已保存到：" << leftHalfPath << std::endl;

    return 0;
}
