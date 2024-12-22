cmake_minimum_required(VERSION 3.10)
project(VoxelCarving)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(VoxelCarving src/detect_markers.cpp)
target_link_libraries(VoxelCarving ${OpenCV_LIBS})