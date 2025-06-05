cmake_minimum_required(VERSION 3.22.1)

project(image_enhance)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)

# 生成图像处理算法程序
add_executable(auto_white_balance autoWhiteBalance.cpp)
add_executable(dark_channel_prior darkChannelPrior.cpp)
add_executable(equalize_enhance equalizeAndEnhance.cpp)

# 链接OpenCV库
target_link_libraries(auto_white_balance PUBLIC ${OpenCV_LIBS})
target_link_libraries(dark_channel_prior PUBLIC ${OpenCV_LIBS})
target_link_libraries(equalize_enhance PUBLIC ${OpenCV_LIBS})
