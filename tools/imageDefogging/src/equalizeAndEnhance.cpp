/*  
 * 文件名：equalizeAndEnhance.cpp
 * 功能：实现图像增强，包括直方图均衡化、增强饱和度
 */
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 读取图像
    Mat image = imread("test.jpg");
    if (image.empty()) {
        cerr << "无法加载图像" << endl;
        return -1;
    }

    // 转换为HSV颜色空间
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // 对V通道进行直方图均衡化
    Mat vChannel;
    vector<Mat> hsvChannels;
    split(hsvImage, hsvChannels);
    equalizeHist(hsvChannels[2], vChannel);
    hsvChannels[2] = vChannel;

    // 合并通道并转换回BGR
    merge(hsvChannels, hsvImage);
    Mat equalizedImage;
    cvtColor(hsvImage, equalizedImage, COLOR_HSV2BGR);

    // 增强饱和度
    Mat enhancedImage;
    hsvChannels[1] = hsvChannels[1] * 1.3; // 增强饱和度，可以根据需求调整
    merge(hsvChannels, hsvImage);
    cvtColor(hsvImage, enhancedImage, COLOR_HSV2BGR);

    // 显示和保存结果
    imshow("Original Image", image);
    imshow("Equalized Image", equalizedImage);
    imshow("Enhanced Image", enhancedImage);

    waitKey(0);
    return 0;
}
