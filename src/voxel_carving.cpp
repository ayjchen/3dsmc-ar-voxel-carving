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

void updateVolume(cv::Mat& rvecs, cv::Mat& tvecs, Volume& vol, int gridSize, float voxelSize) {
    if (rvecs.empty() || tvecs.empty()) {
        std::cerr << "Error: Empty rvecs or tvecs passed to updateVolume!" << std::endl;
        return;
    }

    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

    for (int i = 0; i < rvecs.rows; i++) {
        std::cout << "Rvec " << i << ": " << rvecs.row(i) << std::endl;
        std::cout << "Tvec " << i << ": " << tvecs.row(i) << std::endl;
        for (int j = 0; j < 3; j++) {
            rvec.at<double>(j) += rvecs.at<double>(i, j);
            tvec.at<double>(j) += tvecs.at<double>(i, j);
        }
    }

    rvec /= static_cast<double>(rvecs.rows);
    tvec /= static_cast<double>(tvecs.rows);

    std::cout << "Averaged Rvec: " << rvec.t() << std::endl;
    std::cout << "Averaged Tvec: " << tvec.t() << std::endl;

    // Compute rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    std::cout << "Rotation Matrix: " << R << std::endl;

    // Update volume
    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            for (int z = 0; z < gridSize; ++z) {
                cv::Mat voxelCenter = (cv::Mat_<double>(3, 1) << x * voxelSize, y * voxelSize, z * voxelSize);
                cv::Mat worldCoord = R * voxelCenter + tvec;

                vol.set(x, y, z, worldCoord.at<double>(2));
                // std::cout << "Updating voxel (" << x << ", " << y << ", " << z << ") with value: " << vol.get(x, y, z) << std::endl;
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

        updateVolume(rvecs, tvecs, vol, gridSize, voxelSize);
    }

    SimpleMesh mesh;
    std::cout << "Volume: " << vol.getDimX() << " " << vol.getDimY() << " " << vol.getDimZ() << " " << vol.getData() << std::endl;
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
