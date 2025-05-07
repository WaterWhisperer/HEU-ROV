/*  
文件名：Mouse-DistanceAndDepth.cpp  
功能：
    Depth Measurement窗口：通过鼠标左键拖动框选后，计算框选区域的深度。
    Distance Measurement窗口：通过鼠标左键点击两点测量实际距离。
*/
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// 全局变量
std::vector<cv::Point> box_points;    // 框选坐标 [start, end]
std::vector<cv::Point> meas_points;   // 测量点坐标 [point1, point2]
rs2_intrinsics depth_intrinsics;      // 深度相机内参
std::string current_window;           // 当前活动窗口

// 鼠标事件处理函数
void mouseHandler(int event, int x, int y, int flags, void* param) {
    std::string window_name = *static_cast<std::string*>(param);
    current_window = window_name;

    if (window_name == "Depth Measurement") {
        if (event == cv::EVENT_LBUTTONDOWN) {
            box_points = {cv::Point(x, y)}; // 初始化第一个点
        } else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
            // 拖拽时持续更新第二个点，无论鼠标是否在窗口内
            if (box_points.size() >= 1) {
                if (box_points.size() == 1) box_points.push_back(cv::Point(x, y));
                else box_points[1] = cv::Point(x, y);
            }
        } else if (event == cv::EVENT_LBUTTONUP) {
            if (box_points.size() == 1) box_points.push_back(cv::Point(x, y));
        }
    
    } else if (window_name == "Distance Measurement") {
        if (event == cv::EVENT_LBUTTONDOWN) {
            if (meas_points.size() < 2) {
                meas_points.push_back(cv::Point(x, y));
            } else {
                meas_points = {cv::Point(x, y)};
            }
        }
    }
}

// 计算框选区域深度
float calculateBoxDepth(const std::vector<cv::Point>& box, const rs2::depth_frame& depth_frame) {
    if (box.size() != 2) return 0;

    int x1 = std::min(box[0].x, box[1].x);
    int y1 = std::min(box[0].y, box[1].y);
    int x2 = std::max(box[0].x, box[1].x);
    int y2 = std::max(box[0].y, box[1].y);

    std::vector<float> samples;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_x(x1, x2);
    std::uniform_int_distribution<> dis_y(y1, y2);

    for (int i = 0; i < 50; ++i) {
        int px = dis_x(gen);
        int py = dis_y(gen);
        if (px >= 0 && px < depth_frame.get_width() && py >= 0 && py < depth_frame.get_height()) {
            float dist = depth_frame.get_distance(px, py) * 1000; // 转毫米
            if (dist > 0) samples.push_back(dist);
        }
    }

    if (samples.empty()) return 0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2]; // 中值
}

// 计算三维距离
float calculate3dDistance(const std::vector<cv::Point>& points, const rs2::depth_frame& depth_frame) {
    if (points.size() != 2) return 0;

    std::vector<std::vector<float>> point3d(2);
    for (int i = 0; i < 2; ++i) {
        float depth = depth_frame.get_distance(points[i].x, points[i].y);
        float pixel[2] = {static_cast<float>(points[i].x), static_cast<float>(points[i].y)};
        float point[3];
        rs2_deproject_pixel_to_point(point, &depth_intrinsics, pixel, depth);
        point3d[i] = {point[0], point[1], point[2]};
    }

    float dx = point3d[1][0] - point3d[0][0];
    float dy = point3d[1][1] - point3d[0][1];
    float dz = point3d[1][2] - point3d[0][2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) * 1000; // 毫米
}

int main() {
    // 配置相机
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 60);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 60);
    rs2::pipeline_profile profile = pipe.start(cfg);

    // 获取深度内参
    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    depth_intrinsics = depth_stream.get_intrinsics();

    // 创建窗口并设置鼠标回调
    cv::namedWindow("Depth Measurement");
    cv::namedWindow("Distance Measurement");
    std::string depth_win = "Depth Measurement";
    std::string dist_win = "Distance Measurement";
    cv::setMouseCallback("Depth Measurement", mouseHandler, &depth_win);
    cv::setMouseCallback("Distance Measurement", mouseHandler, &dist_win);

    try {
        while (true) {
            // 获取帧数据
            rs2::frameset frames = pipe.wait_for_frames();
            rs2::depth_frame depth_frame = frames.get_depth_frame();
            rs2::video_frame color_frame = frames.get_color_frame();
            if (!depth_frame || !color_frame) continue;

            // 转换为 OpenCV Mat
            cv::Mat depth_image(cv::Size(640, 480), CV_16U, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat color_image(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat box_img = color_image.clone();
            cv::Mat point_img = color_image.clone();

            // 框选窗口处理
            if (!box_points.empty()) {
                // 实时绘制拖拽框（无论窗口是否激活）
                if (box_points.size() == 1) {
                    // 拖拽中：用临时点绘制动态框
                    cv::Point temp_point = current_window == "Depth Measurement" ? box_points[0] : box_points[0];
                    cv::rectangle(box_img, box_points[0], temp_point, cv::Scalar(0, 200, 0), 2);
                } else if (box_points.size() == 2) {
                    // 绘制最终框和深度值
                    cv::rectangle(box_img, box_points[0], box_points[1], cv::Scalar(0, 255, 0), 2);
                    float depth = calculateBoxDepth(box_points, depth_frame);
                    std::string text = "Depth: " + std::to_string(static_cast<int>(depth + 0.5)) + "mm";
                    cv::putText(box_img, text, cv::Point(box_points[0].x, box_points[0].y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                }
            }           

            // 点测窗口处理
            if (!meas_points.empty()) {
                for (const auto& pt : meas_points) {
                    cv::circle(point_img, pt, 5, cv::Scalar(50, 50, 255), -1);
                }
                if (meas_points.size() == 2) {
                    cv::line(point_img, meas_points[0], meas_points[1], cv::Scalar(255, 100, 100), 2);
                    float distance = calculate3dDistance(meas_points, depth_frame);
                    int mid_x = (meas_points[0].x + meas_points[1].x) / 2;
                    int mid_y = (meas_points[0].y + meas_points[1].y) / 2;
                    std::string text = "Distance: " + std::to_string(static_cast<int>(distance + 0.5)) + "mm";
                    cv::putText(point_img, text, cv::Point(mid_x, mid_y),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
                }
            }

            // 显示图像
            cv::imshow("Depth Measurement", box_img);
            cv::imshow("Distance Measurement", point_img);

            // 退出控制
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
        }
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
    }

    pipe.stop();
    cv::destroyAllWindows();
    return 0;
}