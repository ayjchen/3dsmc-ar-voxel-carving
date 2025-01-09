#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <yaml-cpp/yaml.h>
#include "detect_markers.h"
#include "marching_cubes.h"

namespace fs = std::filesystem;

void loadCalibrationResults(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
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

void saveVoxelsAsOFF(const std::vector<cv::Point3f>& vertices, const std::vector<std::vector<int>>& faces, const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    ofs << "OFF\n";
    ofs << vertices.size() << " " << faces.size() << " 0\n";
    for (const auto& vertex : vertices) {
        ofs << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
    }
    for (const auto& face : faces) {
        ofs << face.size();
        for (const auto& index : face) {
            ofs << " " << index;
        }
        ofs << "\n";
    }

    ofs.close();
}

void calculateExtrinsics(const std::vector<std::vector<cv::Point2f>>& markerCorners, 
                         const std::vector<int>& markerIds, 
                         const cv::Mat& cameraMatrix, 
                         const cv::Mat& distCoeffs, 
                         cv::Mat& rvec, cv::Mat& tvec) {
    // Assume all markers are of the same size
    float markerLength = 0.03f; // Marker size in meters

    // Define 3D object points for a single marker (in marker's local coordinate system)
    std::vector<cv::Point3f> singleMarkerObjectPoints = {
        {0.0f, 0.0f, 0.0f},                           // Top-left corner
        {markerLength, 0.0f, 0.0f},                   // Top-right corner
        {markerLength, markerLength, 0.0f},           // Bottom-right corner
        {0.0f, markerLength, 0.0f}                    // Bottom-left corner
    };

    // Aggregate object points and corresponding image points for all detected markers
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    for (size_t i = 0; i < markerCorners.size(); ++i) {
        for (size_t j = 0; j < singleMarkerObjectPoints.size(); ++j) {
            objectPoints.push_back(singleMarkerObjectPoints[j]);
            imagePoints.push_back(markerCorners[i][j]);
        }
    }

    // SolvePnP to calculate rvec and tvec
    if (!objectPoints.empty() && !imagePoints.empty()) {
        cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    } else {
        std::cerr << "No markers detected or insufficient data for solvePnP." << std::endl;
    }
}


void updateScalarField(const cv::Mat& rvec, const cv::Mat& tvec, cv::Mat& scalarField, const int gridSize, const float voxelSize) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // std::cout << "Scalar Field Size: " << scalarField.size() << std::endl;
    // std::cout << "Scalar Field Rows: " << scalarField.rows << ", Cols: " << scalarField.cols << std::endl;

    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            for (int z = 0; z < gridSize; ++z) {
                cv::Point3f voxelCenter((x - gridSize / 2) * voxelSize, 
                                        (y - gridSize / 2) * voxelSize, 
                                        (z - gridSize / 2) * voxelSize);

                // Transform voxel center to camera coordinates
                cv::Mat voxelMat = (cv::Mat_<double>(3, 1) << voxelCenter.x, voxelCenter.y, voxelCenter.z);
                voxelMat = R * voxelMat + tvec;

                // Check if the voxel projects to valid image coordinates
                if (voxelMat.at<double>(2) > 0) { // Check if in front of the camera
                    cv::Point2f imgPoint;
                    imgPoint.x = voxelMat.at<double>(0) / voxelMat.at<double>(2);
                    imgPoint.y = voxelMat.at<double>(1) / voxelMat.at<double>(2);
                    // std::cout << "imgPoint: " << imgPoint << std::endl;

                    // Check that image point is within the image boundaries
                    if (imgPoint.x >= 0 && imgPoint.x < scalarField.size[0] &&
                        imgPoint.y >= 0 && imgPoint.y < scalarField.size[1]) {
                        scalarField.at<float>(x, y, z) -= 1; // Increment carving effect
                        // std::cout << "Scalar field value: " << scalarField.at<float>(x, y, z) << std::endl;
                    }
                }
            }
        }
    }
}

void performVoxelCarving(const std::vector<cv::Mat>& images, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const std::string& outputFilename) {
    const int gridSize = 100;
    const float voxelSize = 0.01f; // Size of each voxel in meters
    int sizes[3] = {gridSize, gridSize, gridSize};
    cv::Mat scalarField(3, sizes, CV_32F, cv::Scalar(1));

    for (const auto& image : images) {
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        detectArucoMarkers(image, markerCorners, markerIds);

        if (markerIds.empty()) {
            continue;
        }

        cv::Mat rvec, tvec;
        calculateExtrinsics(markerCorners, markerIds, cameraMatrix, distCoeffs, rvec, tvec);
        updateScalarField(rvec, tvec, scalarField, gridSize, voxelSize);
    }

    std::vector<cv::Point3f> vertices;
    std::vector<std::vector<int>> faces;
    std::cout << "scalarField size: " << scalarField.size << std::endl;
    // for (int i = 40; i < 50; i++) {
    //     for (int j = 40; j < 50; j++) {
    //         // for (int k = 0; k < 10; k++) {
    //             std::cout << scalarField.at<float>(i, j, 50) << " ";
    //         // }
    //         // std::cout << std::endl;
    //     }
    // }
    marchingCubes(scalarField, 0.5f, vertices, faces);
    std::cout << "Number of vertices: " << vertices.size() << std::endl;
    std::cout << "Number of faces: " << faces.size() << std::endl;

    saveVoxelsAsOFF(vertices, faces, outputFilename);
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
    for (const auto& entry : fs::directory_iterator(imageDirectory)) {
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
