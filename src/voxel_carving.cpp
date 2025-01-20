#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <yaml-cpp/yaml.h>
#include "detect_markers.h"
#include "marching_cubes.h"
#include "MarchingCubes.h"
#include "SimpleMesh.h"
#include "Volume.h"

namespace fs = std::filesystem;

void loadCalibrationResults( std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    YAML::Node config = YAML::LoadFile(filename);

    // Read camera matrix
    YAML::Node camMatNode = config["camera_matrix"];
    int rows = camMatNode["rows"].as<int>();
    int cols = camMatNode["cols"].as<int>();
    std::vector<double> camData = camMatNode["data"].as<std::vector<double>>();
    cameraMatrix = cv::Mat(rows, cols, CV_64F, camData.data()).clone();

    // Read distortion coefficients
    YAML::Node distNode = config["dist_coeffs"];
    rows = distNode["rows"].as<int>();
    cols = distNode["cols"].as<int>();
    std::vector<double> distData = distNode["data"].as<std::vector<double>>();
    distCoeffs = cv::Mat(rows, cols, CV_64F, distData.data()).clone();

    std::cout << "Camera Matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Distortion Coefficients:\n" << distCoeffs << std::endl;
}

void updateVolume(const cv::Mat& rvecs, const cv::Mat& tvecs, const cv::Mat& image, const cv::Mat& cameraMatrix, 
                  Volume& vol, int gridSize, float voxelSize) {
    if (rvecs.empty() || tvecs.empty()) {
        std::cerr << "Error: Empty rvecs or tvecs passed to updateVolume!" << std::endl;
        return;
    }

    // Convert image to grayscale and create a binary mask where non-black pixels are white
    cv::Mat grayImage, mask;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::threshold(grayImage, mask, 1, 255, cv::THRESH_BINARY);

    for (int poseIdx = 0; poseIdx < rvecs.rows; ++poseIdx) {
        // Convert rvecs and tvecs from Vec3d to Mat
        cv::Mat rvec = cv::Mat(rvecs.at<cv::Vec3d>(poseIdx, 0)).reshape(1, 3); // 3x1 column vector
        cv::Mat tvec = cv::Mat(tvecs.at<cv::Vec3d>(poseIdx, 0)).reshape(1, 3); // 3x1 column vector

        // Compute the rotation matrix
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        // Compute the projection matrix P = K[R|t]
        cv::Mat Rt;
        cv::hconcat(R, tvec, Rt);
        cv::Mat P = cameraMatrix * Rt;

        // Update volume
        for (int x = 0; x < gridSize; ++x) {
            for (int y = 0; y < gridSize; ++y) {
                for (int z = 0; z < gridSize; ++z) {
                    // Compute world coordinates of the voxel center
                    cv::Mat voxelCenter = (cv::Mat_<double>(4, 1) << x * voxelSize, y * voxelSize, z * voxelSize, 1);
                    cv::Mat projected = P * voxelCenter; // Project to the image plane

                    // Convert to 2D pixel coordinates
                    double w = projected.at<double>(2); // Depth
                    if (w <= 0.0) continue; // Skip voxels behind the camera

                    double u = projected.at<double>(0) / w;
                    double v = projected.at<double>(1) / w;

                    // Check if the voxel projection lies within the image boundaries
                    if (u >= 0 && u < mask.cols && v >= 0 && v < mask.rows) {
                        int imgX = static_cast<int>(u);
                        int imgY = static_cast<int>(v);

                        // Check the foreground mask
                        if (mask.at<uchar>(imgY, imgX) > 0) {
                            vol.set(x, y, z, w);
                        } else {
                            vol.set(x, y, z, 0.0);
                        }
                        // std::cout << "Voxel: " << x << " " << y << " " << z << " " << vol.get(x, y, z) << std::endl;
                    }
                }
            }
        }
    }
}

void performVoxelCarving(const std::vector<cv::Mat>& images, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const std::string& outputFilename) {
    const int gridSize = 100;
    const float voxelSize = 0.01f;
    Volume vol(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), gridSize, gridSize, gridSize, 1);

    for (const auto& image : images) {
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        detectArucoMarkers(image, markerCorners, markerIds);

        if (markerIds.empty()) {
            std::cout << "No markers detected in image." << std::endl;
            continue;
        }

        cv::Mat rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.03, cameraMatrix, distCoeffs, rvecs, tvecs);

        // Debug marker pose estimates
        if (rvecs.empty() || tvecs.empty()) {
            std::cerr << "Error: Pose estimation failed. Empty rvecs/tvecs." << std::endl;
            continue;
        }

        std::cout << "Rvecs Type: " << rvecs.type() << ", Size: " << rvecs.size() << std::endl;
        std::cout << "Tvecs Type: " << tvecs.type() << ", Size: " << tvecs.size() << std::endl;

        updateVolume(rvecs, tvecs, image, cameraMatrix, vol, gridSize, voxelSize);
    }

    SimpleMesh mesh;
    std::cout << "Volume: " << vol.getDimX() << " " << vol.getDimY() << " " << vol.getDimZ() << " " << vol.getData() << std::endl;
    double minValue, maxValue;
    vol.computeMinMaxValues(minValue, maxValue);
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            for (int k = 0; k < gridSize; k++) {
                if (vol.get(i, j, k) != 0.0) { 
                    std::cout << vol.get(i, j, k) << " ";
                }
            }
        }
    }
    std::cout << "Volume Min: " << minValue << ", Max: " << maxValue << std::endl;

    marchingCubes(vol, 0.0f, mesh);

    std::cout << "Number of vertices: " << mesh.GetVertices().size() << std::endl;
    std::cout << "Number of faces: " << mesh.GetTriangles().size() << std::endl;

    mesh.WriteMesh(outputFilename);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <calibration_results.yml> <output.off>" << std::endl;
        return -1;
    }

    std::string imageDirectory = argv[1];
    std::string calibrationFile = argv[2];
    std::string outputFilename = argv[3];

    cv::Mat cameraMatrix, distCoeffs;
    loadCalibrationResults(calibrationFile, cameraMatrix, distCoeffs);

    std::vector<cv::Mat> images;
    for ( auto& entry : fs::directory_iterator(imageDirectory)) {
        cv::Mat image = cv::imread(entry.path().string());
        if (!image.empty()) {
            images.push_back(image);
        }
    }

    if (images.empty()) {
        std::cerr << "No valid images found in directory: " << imageDirectory << std::endl;
        return -1;
    }

    performVoxelCarving(images, cameraMatrix, distCoeffs, outputFilename);

    return 0;
}