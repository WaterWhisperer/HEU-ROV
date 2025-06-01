/*
文件名：3D-Structure-Measure.cpp
功能：使用双目相机测量3D结构物的长度、宽度和高度。
使用方法：
    Click 2 points to measure distance.
    After selecting 2 points:
        Press 'l' for Length
        Press 'w' for Width
        Press 'h' for Height
    Press 'c' to clear points.
    Press 'x' to clear all dimensions.
    Press 'p' to plot 3D (need L,W,H).
    Press 'm' to toggle manual display.
    Press 'q' to quit.
*/
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <opencv2/viz.hpp>      // Include OpenCV Viz module for 3D plotting

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <iomanip> // For std::setprecision
#include <map>
#include <optional> // For std::optional (C++17)

// --- Configuration ---
const cv::Scalar LINE_COLOR(0, 255, 0);    // Green
const cv::Scalar POINT_COLOR(255, 0, 0);   // Blue (OpenCV uses BGR)
const cv::Scalar TEXT_COLOR(255, 255, 255); // White
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const double FONT_SCALE = 0.6;
const int FONT_THICKNESS = 1;

// --- Global Variables & Structs ---
bool show_help = true; // 初始显示帮助信息

struct CallbackData {
    std::vector<cv::Point2f> selected_points_2d;
    std::vector<rs2::vertex> current_measurement_3d_points;
    rs2_intrinsics intrinsics;
    rs2::depth_frame depth_frame = rs2::frame{nullptr};
    float depth_scale = 0.0f;
};

CallbackData app_data;
std::map<std::string, std::optional<double>> measurements_log = {
    {"Length", std::nullopt},
    {"Width", std::nullopt},
    {"Height", std::nullopt}
};
const std::string WINDOW_NAME = "ROV Measurement - D455 (C++)";

// --- Helper Functions ---
rs2::vertex deproject_pixel_to_point_custom(const cv::Point2f& pixel, float depth, const rs2_intrinsics& intrinsics_obj) {
    float point[3];
    float pix_arr[2] = {pixel.x, pixel.y};
    rs2_deproject_pixel_to_point(point, &intrinsics_obj, pix_arr, depth);
    return {point[0], point[1], point[2]};
}

double calculate_3d_distance(const rs2::vertex& p1, const rs2::vertex& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) +
                     std::pow(p1.y - p2.y, 2) +
                     std::pow(p1.z - p2.z, 2));
}

void mouse_callback(int event, int x, int y, int flags, void* userdata) {
    CallbackData* data = static_cast<CallbackData*>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (data->depth_frame && data->depth_scale > 0) {
            if (data->selected_points_2d.size() < 2) {
                float depth_value = data->depth_frame.get_distance(x, y);
                if (depth_value == 0) {
                    std::cout << "Warning: Selected point (" << x << ", " << y << ") depth is 0. Point not added. Try another point." << std::endl;
                    return;
                }
                cv::Point2f p2d(static_cast<float>(x), static_cast<float>(y));
                data->selected_points_2d.push_back(p2d);
                rs2::vertex point_3d = deproject_pixel_to_point_custom(p2d, depth_value, data->intrinsics);
                data->current_measurement_3d_points.push_back(point_3d);
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "Selected point " << data->selected_points_2d.size()
                          << ": 2D=(" << x << "," << y << "), 3D=("
                          << point_3d.x << ", " << point_3d.y << ", " << point_3d.z << ")m" << std::endl;
                if (data->selected_points_2d.size() == 2) {
                    double dist = calculate_3d_distance(data->current_measurement_3d_points[0], data->current_measurement_3d_points[1]);
                    std::cout << "Distance between points: " << dist << " meters" << std::endl;
                    std::cout << "Press 'l' for Length, 'w' for Width, 'h' for Height, or 'c' to clear." << std::endl;
                }
            } else {
                std::cout << "2 points already selected. Press 'l'/'w'/'h' to save, or 'c' to clear." << std::endl;
            }
        } else {
            std::cout << "Depth frame or intrinsics not ready yet." << std::endl;
        }
    }
}

void draw_ui(cv::Mat& image, CallbackData& data) {
    int y_offset = 30;
    std::vector<std::string> instructions = {
        "Click 2 points to measure distance.", "After selecting 2 points:",
        "  Press 'l' for Length", "  Press 'w' for Width", "  Press 'h' for Height",
        "Press 'c' to clear points.", "Press 'x' to clear all dimensions.",
        "Press 'p' to plot 3D (need L,W,H).", "Press 'm' to toggle manual display.",
        "Press 'q' to quit."
    };
    
    // 根据show_help决定是否显示完整帮助
    if (show_help) {
        for (size_t i = 0; i < instructions.size(); ++i) {
            cv::putText(image, instructions[i], cv::Point(10, y_offset + i * 20), FONT_FACE, FONT_SCALE * 0.8, TEXT_COLOR, FONT_THICKNESS, cv::LINE_AA);
        }
    } else {
        // 只显示最小指令
        cv::putText(image, "Press 'm' for manual", cv::Point(10, y_offset),
                    FONT_FACE, FONT_SCALE * 0.8, TEXT_COLOR, FONT_THICKNESS, cv::LINE_AA);
    }
    
    int log_y_start = show_help ? (y_offset + static_cast<int>(instructions.size()) * 20 + 10) : (y_offset + 30);
    
    // 绘制点和文本（带边界检查）
    for (size_t i = 0; i < data.selected_points_2d.size(); ++i) {
        cv::circle(image, data.selected_points_2d[i], 3, POINT_COLOR, -1, cv::LINE_AA);
        const auto& p3d = data.current_measurement_3d_points[i];
        std::ostringstream p_text;
        p_text << std::fixed << std::setprecision(2) << "P" << i + 1 << " (" << p3d.x << ", " << p3d.y << ", " << p3d.z << ")m";
        
        // 点坐标文本边界检查
        int text_x = static_cast<int>(data.selected_points_2d[i].x) + 10;
        int text_y = static_cast<int>(data.selected_points_2d[i].y) - 10;
        
        // 计算文本尺寸
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(p_text.str(), FONT_FACE, 0.5, FONT_THICKNESS, &baseline);
        
        // 检查右边界
        if (text_x + text_size.width > image.cols) {
            text_x = static_cast<int>(data.selected_points_2d[i].x) - text_size.width - 10;
        }
        
        // 检查上边界
        if (text_y - text_size.height < 0) {
            text_y = static_cast<int>(data.selected_points_2d[i].y) + text_size.height + 10;
        }
        
        cv::putText(image, p_text.str(), cv::Point(text_x, text_y),
                    FONT_FACE, 0.5, POINT_COLOR, FONT_THICKNESS, cv::LINE_AA);
    }
    
    if (data.selected_points_2d.size() == 2) {
        cv::line(image, data.selected_points_2d[0], data.selected_points_2d[1], LINE_COLOR, 1, cv::LINE_AA);
        double dist = calculate_3d_distance(data.current_measurement_3d_points[0], data.current_measurement_3d_points[1]);
        cv::Point mid_point((data.selected_points_2d[0].x + data.selected_points_2d[1].x) / 2,
                             (data.selected_points_2d[0].y + data.selected_points_2d[1].y) / 2);
        std::ostringstream dist_text;
        dist_text << std::fixed << std::setprecision(3) << dist << " m";
        
        // 距离文本边界检查
        int text_x = mid_point.x;
        int text_y = mid_point.y - 10;
        
        // 计算文本尺寸
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(dist_text.str(), FONT_FACE, FONT_SCALE, FONT_THICKNESS, &baseline);
        
        // 检查右边界
        if (text_x + text_size.width/2 > image.cols) {
            text_x = image.cols - text_size.width - 5;
        }
        // 检查左边界
        else if (text_x - text_size.width/2 < 0) {
            text_x = text_size.width/2 + 5;
        }
        
        // 检查上边界
        if (text_y - text_size.height < 0) {
            text_y = mid_point.y + text_size.height + 10;
        }
        
        cv::putText(image, dist_text.str(), cv::Point(text_x, text_y), 
                    FONT_FACE, FONT_SCALE, LINE_COLOR, FONT_THICKNESS, cv::LINE_AA);
    }
    
    cv::putText(image, "Saved Dimensions:", cv::Point(10, log_y_start), FONT_FACE, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv::LINE_AA);
    int i = 0;
    for (const auto& pair : measurements_log) {
        std::ostringstream dim_text;
        dim_text << "- " << pair.first << ": ";
        if (pair.second.has_value()) {
            dim_text << std::fixed << std::setprecision(3) << pair.second.value() << " m";
        } else {
            dim_text << "Not Set";
        }
        cv::putText(image, dim_text.str(), cv::Point(10, log_y_start + 20 + i * 25), FONT_FACE, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv::LINE_AA);
        i++;
    }
}

void plot_3d_structure(const std::optional<double>& length_opt, const std::optional<double>& width_opt, const std::optional<double>& height_opt) {
    if (!length_opt.has_value() || !width_opt.has_value() || !height_opt.has_value()) {
        std::cout << "Error: Need to set Length, Width, and Height to plot 3D structure." << std::endl;
        return;
    }
    double L = length_opt.value();
    double W = width_opt.value();
    double H = height_opt.value();

    cv::viz::Viz3d viz_window("Measured 3D Structure");
    viz_window.setBackgroundColor(cv::viz::Color::white());
    viz_window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

    cv::Point3d cube_center(L / 2.0, W / 2.0, H / 2.0);
    cv::Point3d cube_size(L, W, H);
    cv::viz::WCube cube_widget(cube_center, cube_size, true, cv::viz::Color::cyan());
    cube_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    viz_window.showWidget("Cube", cube_widget);
    
    std::vector<cv::Point3d> p = {
        {0,0,0}, {L,0,0}, {L,W,0}, {0,W,0},
        {0,0,H}, {L,0,H}, {L,W,H}, {0,W,H}
    };
    viz_window.showWidget("L01", cv::viz::WLine(p[0], p[1], cv::viz::Color::red()));
    viz_window.showWidget("L12", cv::viz::WLine(p[1], p[2], cv::viz::Color::red()));
    viz_window.showWidget("L23", cv::viz::WLine(p[2], p[3], cv::viz::Color::red()));
    viz_window.showWidget("L30", cv::viz::WLine(p[3], p[0], cv::viz::Color::red()));
    viz_window.showWidget("L45", cv::viz::WLine(p[4], p[5], cv::viz::Color::red()));
    viz_window.showWidget("L56", cv::viz::WLine(p[5], p[6], cv::viz::Color::red()));
    viz_window.showWidget("L67", cv::viz::WLine(p[6], p[7], cv::viz::Color::red()));
    viz_window.showWidget("L74", cv::viz::WLine(p[7], p[4], cv::viz::Color::red()));
    viz_window.showWidget("L04", cv::viz::WLine(p[0], p[4], cv::viz::Color::red()));
    viz_window.showWidget("L15", cv::viz::WLine(p[1], p[5], cv::viz::Color::red()));
    viz_window.showWidget("L26", cv::viz::WLine(p[2], p[6], cv::viz::Color::red()));
    viz_window.showWidget("L37", cv::viz::WLine(p[3], p[7], cv::viz::Color::red()));

    double text_scale = std::max({L, W, H}) / 20.0; 
    double offset_factor = 0.15; 

    std::ostringstream l_text; l_text << std::fixed << std::setprecision(3) << "L: " << L << "m";
    cv::viz::WText3D length_label(l_text.str(), cv::Point3d(L / 2.0, -W * offset_factor, 0), text_scale, false, cv::viz::Color::black());
    viz_window.showWidget("LengthLabel", length_label);
    viz_window.showWidget("LengthLine", cv::viz::WLine(cv::Point3d(0, -W*offset_factor*0.8, 0), cv::Point3d(L, -W*offset_factor*0.8, 0), cv::viz::Color::black()));

    std::ostringstream w_text; w_text << std::fixed << std::setprecision(3) << "W: " << W << "m";
    cv::viz::WText3D width_label(w_text.str(), cv::Point3d(-L * offset_factor, W / 2.0, 0), text_scale, false, cv::viz::Color::black());
    viz_window.showWidget("WidthLabel", width_label);
    viz_window.showWidget("WidthLine", cv::viz::WLine(cv::Point3d(-L*offset_factor*0.8, 0, 0), cv::Point3d(-L*offset_factor*0.8, W, 0), cv::viz::Color::black()));

    std::ostringstream h_text; h_text << std::fixed << std::setprecision(3) << "H: " << H << "m";
    cv::viz::WText3D height_label(h_text.str(), cv::Point3d(-L * offset_factor, -W * offset_factor, H / 2.0), text_scale, false, cv::viz::Color::black());
    viz_window.showWidget("HeightLabel", height_label);
    viz_window.showWidget("HeightLine", cv::viz::WLine(cv::Point3d(-L*offset_factor*0.8, -W*offset_factor*0.8, 0), cv::Point3d(-L*offset_factor*0.8, -W*offset_factor*0.8, H), cv::viz::Color::black()));
    
    // --- Corrected Camera View Setting ---
    // Define camera position, focal point (look-at point), and up vector for the world
    cv::Vec3d cam_pos(L * 1.5, W * -0.5, H * 1.5);    // Position the camera
    cv::Vec3d cam_lookat(L / 2.0, W / 2.0, H / 2.0); // Look at the center of the cuboid
    cv::Vec3d cam_up(0.0, 0.0, 1.0);                 // Z-axis is "up" in the world

    // Create the camera pose
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_lookat, cam_up);
    viz_window.setViewerPose(cam_pose); // Set the viewer pose

    std::cout << "Displaying 3D structure. Close the viz window to continue..." << std::endl;
    viz_window.spin(); 
}


int main() {
    rs2::pipeline pipe;
    rs2::config cfg;

    int desired_width1 = 848, desired_height1 = 480;
    int desired_width2 = 1280, desired_height2 = 720; 
    int desired_fps = 30;

    rs2::video_stream_profile depth_profile_selected; 
    rs2::video_stream_profile color_profile_selected;

    try {
        auto profile = cfg.resolve(pipe);
        auto dev = profile.get_device();
        std::cout << "Device: " << dev.get_info(RS2_CAMERA_INFO_NAME) << std::endl;

        auto depth_sensor = dev.first<rs2::depth_sensor>();
        if (depth_sensor.supports(RS2_OPTION_VISUAL_PRESET)) {
            try {
                // --- Corrected Visual Preset Setting ---
                // RS2_VISUAL_PRESET_HIGH_ACCURACY is typically enum value 3.
                // The set_option function takes a float.
                float preset_value = 3.0f; // Value for High Accuracy
                // Check if the specific macro RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY is available
                // (as suggested by a previous compiler error). If so, it's safer.
                // If not, using 3.0f is the standard.
                #ifdef RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY
                    std::cout << "Using RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY." << std::endl;
                    preset_value = static_cast<float>(RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);
                #else
                    // Check if the generic RS2_VISUAL_PRESET_HIGH_ACCURACY macro is available
                    #ifdef RS2_VISUAL_PRESET_HIGH_ACCURACY
                         std::cout << "Using RS2_VISUAL_PRESET_HIGH_ACCURACY macro." << std::endl;
                         preset_value = static_cast<float>(RS2_VISUAL_PRESET_HIGH_ACCURACY);
                    #else
                         std::cout << "Using hardcoded preset value 3.0f for High Accuracy." << std::endl;
                         // preset_value is already 3.0f
                    #endif
                #endif
                depth_sensor.set_option(RS2_OPTION_VISUAL_PRESET, preset_value);
                std::cout << "Attempted to set visual preset to High Accuracy (value: " << preset_value << ")." << std::endl;
            } catch (const rs2::error& e) {
                std::cerr << "Failed to set high accuracy preset: " << e.what() << std::endl;
            }
        } else {
            std::cout << "Device does not support visual preset option." << std::endl;
        }
        
        auto stream_profiles = depth_sensor.get_stream_profiles();
        for (const auto& p_loop : stream_profiles) { // Renamed p to p_loop to avoid conflict
            auto vp = p_loop.as<rs2::video_stream_profile>();
            if (vp && vp.format() == RS2_FORMAT_Z16 && vp.fps() == desired_fps) {
                if (vp.width() == desired_width1 && vp.height() == desired_height1) {
                    depth_profile_selected = vp;
                    break;
                }
            }
        }
        if (!depth_profile_selected) {
            for (const auto& p_loop : stream_profiles) {  // Renamed p to p_loop
                auto vp = p_loop.as<rs2::video_stream_profile>();
                if (vp && vp.format() == RS2_FORMAT_Z16 && vp.fps() == desired_fps) {
                    if (vp.width() == desired_width2 && vp.height() == desired_height2) {
                        depth_profile_selected = vp;
                        break;
                    }
                }
            }
        }

        if (!depth_profile_selected) {
            std::cerr << "Error: Could not find a suitable depth stream configuration." << std::endl;
            return EXIT_FAILURE;
        }
         std::cout << "Using Depth Stream: " << depth_profile_selected.width() << "x" << depth_profile_selected.height() 
                  << " @ " << depth_profile_selected.fps() << "fps, format " << depth_profile_selected.format() << std::endl;

        auto color_sensor = dev.first<rs2::color_sensor>();
        stream_profiles = color_sensor.get_stream_profiles(); // Re-assign for color sensor

        for (const auto& p_loop : stream_profiles) { // Renamed p to p_loop
             auto vp = p_loop.as<rs2::video_stream_profile>();
             if (vp && vp.format() == RS2_FORMAT_BGR8 && vp.fps() == desired_fps) {
                 if (vp.width() == depth_profile_selected.width() && vp.height() == depth_profile_selected.height()) {
                     color_profile_selected = vp;
                     break;
                 }
             }
        }
        if (!color_profile_selected) { 
             for (const auto& p_loop : stream_profiles) { // Renamed p to p_loop
                auto vp = p_loop.as<rs2::video_stream_profile>();
                if (vp && vp.format() == RS2_FORMAT_BGR8 && vp.fps() == desired_fps) {
                    if (vp.width() == desired_width1 && vp.height() == desired_height1) { 
                         color_profile_selected = vp;
                         break;
                    }
                }
             }
        }
         if (!color_profile_selected) {
            std::cerr << "Error: Could not find a suitable color stream configuration." << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Using Color Stream: " << color_profile_selected.width() << "x" << color_profile_selected.height() 
                  << " @ " << color_profile_selected.fps() << "fps, format " << color_profile_selected.format() << std::endl;

        cfg.enable_stream(RS2_STREAM_DEPTH, depth_profile_selected.width(), depth_profile_selected.height(), RS2_FORMAT_Z16, desired_fps);
        cfg.enable_stream(RS2_STREAM_COLOR, color_profile_selected.width(), color_profile_selected.height(), RS2_FORMAT_BGR8, desired_fps);

    } catch (const rs2::error & e) {
        std::cerr << "RealSense configuration error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    rs2::pipeline_profile active_profile = pipe.start(cfg);
    auto depth_sensor_active = active_profile.get_device().first<rs2::depth_sensor>();
    app_data.depth_scale = depth_sensor_active.get_depth_scale();
    std::cout << "Depth Scale: " << app_data.depth_scale << std::endl;

    rs2::align align_to_color(RS2_STREAM_COLOR);
    auto stream = active_profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    app_data.intrinsics = stream.get_intrinsics();
    std::cout << "Color Stream Intrinsics (Aligned): fx=" << app_data.intrinsics.fx << ", fy=" << app_data.intrinsics.fy
              << ", ppx=" << app_data.intrinsics.ppx << ", ppy=" << app_data.intrinsics.ppy
              << ", width=" << app_data.intrinsics.width << ", height=" << app_data.intrinsics.height << std::endl;

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(WINDOW_NAME, mouse_callback, &app_data);
    std::cout << "Camera initialized. Click two points on the color image to measure distance." << std::endl;
    std::cout << "Keybindings are shown in the top-left of the window." << std::endl;

    rs2::colorizer color_map; 
    try {
        while (true) {
            rs2::frameset frames = pipe.wait_for_frames();
            rs2::frameset aligned_frames = align_to_color.process(frames);
            rs2::depth_frame rs_depth_frame = aligned_frames.get_depth_frame();
            rs2::video_frame rs_color_frame = aligned_frames.get_color_frame();
            if (!rs_depth_frame || !rs_color_frame) continue;
            app_data.depth_frame = rs_depth_frame; 
            cv::Mat color_image(cv::Size(rs_color_frame.get_width(), rs_color_frame.get_height()), CV_8UC3, (void*)rs_color_frame.get_data(), cv::Mat::AUTO_STEP);
            rs2::frame depth_for_display = rs_depth_frame.apply_filter(color_map);
            cv::Mat depth_colormap(cv::Size(rs_depth_frame.get_width(), rs_depth_frame.get_height()), CV_8UC3, (void*)depth_for_display.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat display_image = color_image.clone();
            draw_ui(display_image, app_data);
            cv::Mat combined_output;
            cv::hconcat(display_image, depth_colormap, combined_output);
            cv::imshow(WINDOW_NAME, combined_output);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
            else if (key == 'c') {
                app_data.selected_points_2d.clear(); app_data.current_measurement_3d_points.clear();
                std::cout << "Current points cleared." << std::endl;
            } else if (key == 'x') {
                measurements_log["Length"] = std::nullopt; measurements_log["Width"] = std::nullopt; measurements_log["Height"] = std::nullopt;
                std::cout << "All saved dimensions cleared." << std::endl;
            } else if (key == 'l' || key == 'w' || key == 'h') {
                if (app_data.current_measurement_3d_points.size() == 2) {
                    double dist = calculate_3d_distance(app_data.current_measurement_3d_points[0], app_data.current_measurement_3d_points[1]);
                    std::string dim_type = (key == 'l' ? "Length" : (key == 'w' ? "Width" : "Height"));
                    measurements_log[dim_type] = dist;
                    std::cout << std::fixed << std::setprecision(3) << "Saved " << dim_type << ": " << dist << " m" << std::endl;
                    app_data.selected_points_2d.clear(); app_data.current_measurement_3d_points.clear();
                } else {
                    std::cout << "Please select 2 points first to define a measurement." << std::endl;
                }
            } else if (key == 'p') {
                 plot_3d_structure(measurements_log["Length"], measurements_log["Width"], measurements_log["Height"]);
            } else if (key == 'm') {
                show_help = !show_help; // 切换帮助信息显示
                std::cout << "Help display " << (show_help ? "enabled" : "disabled") << std::endl;
            }
        }
    } catch (const rs2::error & e) {
        std::cerr << "RealSense runtime error: " << e.what() << std::endl; return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl; return EXIT_FAILURE;
    }
    pipe.stop();
    cv::destroyAllWindows();
    std::cout << "Pipeline stopped, windows closed." << std::endl;
    return EXIT_SUCCESS;
}
