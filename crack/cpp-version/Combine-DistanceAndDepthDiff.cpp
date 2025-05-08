/*  
文件名: Combine-DistanceAndDepthDiff.cpp
功能：深度测量和距离测量的结合，显示在同一窗口中。
    Depth Measurement：左侧视图，用于深度与深度差测量（支持裂缝和平地区域两次框选）。
    Distance Measurement：右侧视图，用于距离测量。
*/
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// 全局变量
std::vector<cv::Point> crack_box;     // 裂缝框选坐标 [start, end]
std::vector<cv::Point> ground_box;    // 平地板选坐标 [start, end]
std::vector<cv::Point> meas_points;   // 测量点坐标 [point1, point2]
rs2_intrinsics depth_intrinsics;      // 深度相机内参
std::string current_window;           // 当前活动窗口
const int WINDOW_WIDTH = 640;         // 单个视图宽度

// 鼠标事件处理函数
void mouseHandler(int event, int x, int y, int flags, void* userdata) {
    static bool is_dragging = false;
    static int current_box = 0; // 0:裂缝 1:平地
    int adj_x;

    // 判断操作区域
    if (!is_dragging) {
        if (x < WINDOW_WIDTH) {
            current_window = "Depth";
            adj_x = x;
        } else {
            current_window = "Distance";
            adj_x = x - WINDOW_WIDTH;
        }
    } else {
        current_window = "Depth";
        adj_x = (x < WINDOW_WIDTH) ? x : WINDOW_WIDTH - 1;
    }

    // 处理左侧视图（Depth）的拖拽框
    if (current_window == "Depth") {
        if (event == cv::EVENT_LBUTTONDOWN) {
            is_dragging = true;
            if (current_box == 0) {
                crack_box.clear();
                crack_box.push_back(cv::Point(adj_x, y));
            } else {
                ground_box.clear();
                ground_box.push_back(cv::Point(adj_x, y));
            }
        } else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
            std::vector<cv::Point>* target = (current_box == 0) ? &crack_box : &ground_box;
            if (target->size() >= 1) {
                if (target->size() == 1) target->push_back(cv::Point(adj_x, y));
                else (*target)[1] = cv::Point(adj_x, y);
            }
        } else if (event == cv::EVENT_LBUTTONUP) {
            is_dragging = false;
            std::vector<cv::Point>* target = (current_box == 0) ? &crack_box : &ground_box;
            if (target->size() == 1) {
                target->push_back(cv::Point(adj_x, y));
                current_box = 1 - current_box; // 切换框类型
            }
        }
    }

    // 处理右侧视图（Distance）的测量点
    else if (current_window == "Distance") {
        if (event == cv::EVENT_LBUTTONDOWN) {
            if (meas_points.size() < 2) {
                meas_points.push_back(cv::Point(adj_x, y));
            } else {
                meas_points = {cv::Point(adj_x, y)};
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
    cv::namedWindow("Measurement System");
    cv::setMouseCallback("Measurement System", mouseHandler);

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
            cv::Mat left_img = color_image.clone();   // 左侧视图
            cv::Mat right_img = color_image.clone();  // 右侧视图

            // 深度测量处理（左侧）
            // 绘制裂缝框
            if (!crack_box.empty()) {
                if (crack_box.size() == 1) {
                    cv::rectangle(left_img, crack_box[0], crack_box[0], cv::Scalar(0, 200, 0), 2);
                } else if (crack_box.size() == 2) {
                    cv::rectangle(left_img, crack_box[0], crack_box[1], cv::Scalar(0, 255, 0), 2);
                    float depth = calculateBoxDepth(crack_box, depth_frame);
                    std::string text = "Crack: " + std::to_string(static_cast<int>(depth + 0.5)) + "mm";
                    cv::putText(left_img, text, crack_box[0] + cv::Point(0, -10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                }
            }

            // 绘制平地块
            if (!ground_box.empty()) {
                if (ground_box.size() == 1) {
                    cv::rectangle(left_img, ground_box[0], ground_box[0], cv::Scalar(200, 0, 0), 2);
                } else if (ground_box.size() == 2) {
                    cv::rectangle(left_img, ground_box[0], ground_box[1], cv::Scalar(255, 0, 0), 2);
                    float depth = calculateBoxDepth(ground_box, depth_frame);
                    std::string text = "Ground: " + std::to_string(static_cast<int>(depth + 0.5)) + "mm";
                    cv::putText(left_img, text, ground_box[0] + cv::Point(0, -10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
                }
            }

            // 计算并显示深度差
            if (crack_box.size() == 2 && ground_box.size() == 2) {
                float crack_depth = calculateBoxDepth(crack_box, depth_frame);
                float ground_depth = calculateBoxDepth(ground_box, depth_frame);
                float depth_diff = ground_depth - crack_depth;
                std::string diff_text = "Depth Diff: " + std::to_string(static_cast<int>(std::abs(depth_diff) + 0.5)) + "mm";
                cv::putText(left_img, diff_text, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }

            // 距离测量处理（右侧）
            if (!meas_points.empty()) {
                for (const auto& pt : meas_points) {
                    cv::circle(right_img, pt, 3, cv::Scalar(50, 50, 255), -1);
                }
                if (meas_points.size() == 2) {
                    cv::line(right_img, meas_points[0], meas_points[1], cv::Scalar(255, 100, 100), 2);
                    float distance = calculate3dDistance(meas_points, depth_frame);
                    int mid_x = (meas_points[0].x + meas_points[1].x) / 2;
                    int mid_y = (meas_points[0].y + meas_points[1].y) / 2;
                    std::string text = "Distance: " + std::to_string(static_cast<int>(distance + 0.5)) + "mm";
                    cv::putText(right_img, text, cv::Point(mid_x, mid_y),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
                }
            }

            // 合并显示
            cv::Mat combined;
            cv::hconcat(left_img, right_img, combined);
            // 添加分割线
            cv::line(combined, cv::Point(WINDOW_WIDTH, 0), cv::Point(WINDOW_WIDTH, 480), cv::Scalar(200, 200, 200), 2);
            // 添加文字标注
            cv::putText(combined, "Depth Measurement", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::putText(combined, "Distance Measurement", cv::Point(WINDOW_WIDTH + 10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

            cv::imshow("Measurement System", combined);

            // 退出控制（添加重置功能示例）
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
            else if (key == 'r') { // 按r键重置所有框
                crack_box.clear();
                ground_box.clear();
                meas_points.clear();
            }
        }
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
    }

    pipe.stop();
    cv::destroyAllWindows();
    return 0;
}