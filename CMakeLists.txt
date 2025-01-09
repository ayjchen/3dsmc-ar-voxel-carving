cmake_minimum_required(VERSION 3.10)
project(VoxelCarving)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# add_executable(CameraCalibration src/camera_calibration.cpp)
add_executable(VoxelCarving src/voxel_carving.cpp src/detect_markers.cpp src/marching_cubes.cpp)
# add_executable(MarchingCubes src/marching_cubes.cpp)

# target_link_libraries(CameraCalibration ${OpenCV_LIBS})
target_link_libraries(VoxelCarving ${OpenCV_LIBS} yaml-cpp)
# target_link_libraries(MarchingCubes ${OpenCV_LIBS})
