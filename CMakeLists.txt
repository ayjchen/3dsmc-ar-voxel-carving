cmake_minimum_required(VERSION 3.10)
project(VoxelCarving)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# # Define header and source files
# set(HEADERS
#     Eigen.h
#     ImplicitSurface.h
#     MarchingCubes.h
#     Volume.h
# )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# add_executable(CameraCalibration src/camera_calibration.cpp)
# add_executable(VoxelCarving src/voxel_carving.cpp src/detect_markers.cpp src/marching_cubes.cpp src/Volume.cpp)
add_executable(Postprocessing src/postprocessing.cpp)

# target_link_libraries(CameraCalibration ${OpenCV_LIBS})
# target_link_libraries(VoxelCarving ${OpenCV_LIBS} yaml-cpp)
target_link_libraries(Postprocessing ${OpenCV_LIBS})
