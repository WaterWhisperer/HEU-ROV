/*
 * 文件名：darkChannelPrior.cpp
 * 功能：计算暗通道图像、估计大气光、计算传输图、恢复图像
 */
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 计算暗通道图像
Mat darkChannel(const Mat& image, int size) {
    Mat bgr[3];
    split(image, bgr);
    Mat minChannel;
    min(bgr[0], bgr[1], minChannel);
    min(minChannel, bgr[2], minChannel);
    Mat darkChannel;
    erode(minChannel, darkChannel, getStructuringElement(MORPH_RECT, Size(size, size)));
    return darkChannel;
}

// 估计大气光
Vec3b estimateAtmosphericLight(const Mat& image, const Mat& darkChannel) {
    Mat darkChannelFlat;
    darkChannel.reshape(1, darkChannel.total()).copyTo(darkChannelFlat);
    Mat sortedIndices;
    sortIdx(darkChannelFlat, sortedIndices, SORT_EVERY_COLUMN + SORT_DESCENDING);
    
    int numPixels = static_cast<int>(0.02 * darkChannel.total()); // 增加到2%
    Vec3b atmosphericLight = {0, 0, 0};
    for (int i = 0; i < numPixels; ++i) {
        int idx = sortedIndices.at<int>(i);
        atmosphericLight[0] += image.at<Vec3b>(idx)[0];
        atmosphericLight[1] += image.at<Vec3b>(idx)[1];
        atmosphericLight[2] += image.at<Vec3b>(idx)[2];
    }
    atmosphericLight[0] /= numPixels;
    atmosphericLight[1] /= numPixels;
    atmosphericLight[2] /= numPixels;
    return atmosphericLight;
}

// 计算传输图
Mat computeTransmission(const Mat& image, const Vec3b& atmosphericLight, int size) {
    Mat transmission(image.size(), CV_32F);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            Vec3b I = image.at<Vec3b>(y, x);
            transmission.at<float>(y, x) = 1.0 - exp(-0.1 * (min(I[0], min(I[1], I[2])) / float(atmosphericLight[0])));
        }
    }
    // 使用高斯模糊平滑传输图
    GaussianBlur(transmission, transmission, Size(5, 5), 0);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(size, size));
    dilate(transmission, transmission, kernel);
    return transmission;
}

// 恢复图像
Mat recoverImage(const Mat& image, const Vec3b& atmosphericLight, const Mat& transmission, float t0) {
    Mat recovered(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            Vec3b I = image.at<Vec3b>(y, x);
            float t = max(transmission.at<float>(y, x), t0);
            for (int c = 0; c < 3; ++c) {
                recovered.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(atmosphericLight[c] + (I[c] - atmosphericLight[c]) / t);
            }
        }
    }
    return recovered;
}

int main() {
    // 读取图像
    Mat image = imread("test.jpg");
    if (image.empty()) {
        cerr << "无法加载图像" << endl;
        return -1;
    }

    // 计算暗通道图像
    Mat darkChannelImg = darkChannel(image, 15);
    
    // 估计大气光
    Vec3b atmosphericLight = estimateAtmosphericLight(image, darkChannelImg);
    
    // 计算传输图
    Mat transmissionImg = computeTransmission(image, atmosphericLight, 15);
    
    // 恢复图像，调整 t0 的值
    Mat recoveredImage = recoverImage(image, atmosphericLight, transmissionImg, 0.25f); // 调整 t0 值

    // 显示和保存结果
    imshow("Original Image", image);
    imshow("Recovered Image", recoveredImage);

    waitKey(0);
    return 0;
}
