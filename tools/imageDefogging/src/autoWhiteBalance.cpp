/*
 * 文件名: autoWhiteBalance.cpp
 * 功能: 实现自动白平衡算法
 * 作者: WaterWhisperer
 * 创建日期: 2024-12-04
 * 更新日期: 2025-06-05
 * 版权: 自由使用，保留作者信息
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>

using namespace cv;
using namespace std;

void motowhitebalance(const Mat& originalImage, Mat& adjustedImage) {
    // 将图像从RGB颜色空间转换到YCbCr颜色空间
    Mat im1;
    cvtColor(originalImage, im1, COLOR_RGB2YCrCb);
    Mat Lu, Cb, Cr;
    split(im1, vector<Mat>{Lu, Cb, Cr});
    
    // 获取图像尺寸
    int x = im1.rows;
    int y = im1.cols;

    // 定义test函数来计算平均值和差值
    auto test = [](const Mat& Cb, const Mat& Cr) {
        Scalar Mb = mean(Cb);
        Scalar Mr = mean(Cr);
        double Db = sum(Cb - Mb[0])[0] / (Cb.rows * Cb.cols);
        double Dr = sum(Cr - Mr[0])[0] / (Cr.rows * Cr.cols);
        return make_tuple(Mb[0], Mr[0], Db, Dr);
    };
    
    // 处理图像块
    Mat I1 = im1(Rect(0, 0, y / 2, x / 2));
    auto [Mb1, Mr1, Db1, Dr1] = test(Cb(Rect(0, 0, y / 2, x / 2)), Cr(Rect(0, 0, y / 2, x / 2)));

    Mat I2 = im1(Rect(0, x / 2, y / 2, x / 2));
    test(Cb(Rect(0, x / 2, y / 2, x / 2)), Cr(Rect(0, x / 2, y / 2, x / 2)));

    Mat I3 = im1(Rect(y / 2, 0, y / 2, x / 2));
    auto [Mb3, Mr3, Db3, Dr3] = test(Cb(Rect(y / 2, 0, y / 2, x / 2)), Cr(Rect(y / 2, 0, y / 2, x / 2)));

    Mat I4 = im1(Rect(y / 2, x / 2, y / 2, x / 2));
    auto [Mb4, Mr4, Db4, Dr4] = test(Cb(Rect(y / 2, x / 2, y / 2, x / 2)), Cr(Rect(y / 2, x / 2, y / 2, x / 2)));

    // 计算整体平均值和差值
    double Mr = (Mr1 + Mr3 + Mr4) / 3;
    double Mb = (Mb1 + Mb3 + Mb4) / 3;
    double Dr = (Dr1 + Dr3 + Dr4) / 3;
    double Db = (Db1 + Db3 + Db4) / 3;

    // 提取亮度值
    vector<uchar> Ciny;
    Mat tst = Mat::zeros(Lu.size(), CV_8UC1);
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            double b1 = Cb.at<uchar>(i, j) - (Mb + Db * (Mb > 0 ? 1 : -1));
            double b2 = Cr.at<uchar>(i, j) - (1.5 * Mr + Dr * (Mr > 0 ? 1 : -1));
            if (abs(b1) < abs(1.5 * Db) && abs(b2) < abs(1.5 * Dr)) {
                Ciny.push_back(Lu.at<uchar>(i, j));
                tst.at<uchar>(i, j) = Lu.at<uchar>(i, j);
            }
        }
    }

    // 排序和选择
    sort(Ciny.begin(), Ciny.end(), greater<uchar>());
    int count = round(Ciny.size() / 10);
    vector<uchar> Ciny2(Ciny.begin(), Ciny.begin() + count);
    uchar mn = *min_element(Ciny2.begin(), Ciny2.end());
    Mat index = Lu > mn;

    // 计算增益
    Mat R, G, B;
    split(originalImage, vector<Mat>{B, G, R}); // OpenCV 使用 BGR 格式

    // 确保通道不为空
    if (R.empty() || G.empty() || B.empty()) {
        throw runtime_error("Error: One of the color channels is empty.");
    }

    Scalar Rave = mean(R, index);
    Scalar Gave = mean(G, index);
    Scalar Bave = mean(B, index);
    uchar Ymax = *max_element(Lu.begin<uchar>(), Lu.end<uchar>());

    double Rgain = static_cast<double>(Ymax) / Rave[0];
    double Ggain = static_cast<double>(Ymax) / Gave[0];
    double Bgain = static_cast<double>(Ymax) / Bave[0];

    // 调整图像
    R = R * Rgain;
    G = G * Ggain;
    B = B * Bgain;

    // 合并通道
    vector<Mat> channels = {B, G, R}; // OpenCV 使用 BGR 格式
    merge(channels, adjustedImage);
}

int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <输入图像路径>" << std::endl;
        return -1;
    }
    
    // 读取图像
    Mat originalImage = imread(argv[1]);
    if (originalImage.empty()) {
        std::cerr << "错误: 无法读取图像: " << argv[1] << std::endl;
        return -1;
    }
    
    // 应用白平衡调整
    Mat adjustedImage;
    motowhitebalance(originalImage, adjustedImage);
    
    // 显示图像
    imshow("Original Image", originalImage);
    imshow("Adjusted Image", adjustedImage);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
