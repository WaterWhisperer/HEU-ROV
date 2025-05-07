# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(CombineDistanceAndDepth)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${realsense2_INCLUDE_DIRS})

# 添加可执行文件
add_executable(DistanceAndDepth Combine-DistanceAndDepth.cpp)

# 链接库文件
target_link_libraries(DistanceAndDepth ${OpenCV_LIBS} ${realsense2_LIBRARY})