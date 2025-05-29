cmake_minimum_required(VERSION 3.10)
project(3D_Structure_Measure_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find librealsense2
find_package(realsense2 REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc viz) # viz is needed for 3D plotting

include_directories(
    ${realsense2_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} 3D-Structure-Measure.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    realsense2::realsense2
    ${OpenCV_LIBS}
)
