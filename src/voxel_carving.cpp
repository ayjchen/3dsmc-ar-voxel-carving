#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
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
                  Volume& vol, int gridSize, float voxelSize, int imageIndex) {
    if (rvecs.empty() || tvecs.empty()) {
        std::cerr << "Error: Empty rvecs or tvecs passed to updateVolume!" << std::endl;
        return;
    }

    // Convert image to grayscale and create a binary mask where non-black pixels are white
    cv::Mat grayImage, mask;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::threshold(grayImage, mask, 1, 255, cv::THRESH_BINARY);

    // Compute the camera projection matrix P = K[R|t]
    cv::Mat R;
    cv::Rodrigues(rvecs, R);
    // cv::Mat translation = -R.t() * tvecs;
    cv::Mat Rt;
    cv::hconcat(R, tvecs, Rt);
    cv::Mat P = cameraMatrix * Rt;

    // Update volume by translating the voxel centers to the center of the volume
    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            for (int z = 0; z < gridSize; ++z) {
                // Compute world coordinates of the voxel center, adjusting to the grid center
                cv::Mat voxelCenter = (cv::Mat_<double>(4, 1) << (x - gridSize / 2) * voxelSize, 
                                                        (y - gridSize / 2) * voxelSize, 
                                                        (z - gridSize / 2) * voxelSize, 1);
                cv::Mat projected = P * voxelCenter; // Project to the image plane

                // Convert to 2D pixel coordinates
                double w = projected.at<double>(2); // Depth
                if (w <= 0.0) continue; // Skip voxels behind the camera

                double u = projected.at<double>(0) / w;
                double v = projected.at<double>(1) / w;

                int imgX = static_cast<int>(u);
                int imgY = static_cast<int>(v);

                // Check if the voxel projection lies within the image boundaries
                if (u >= 0 && u < mask.cols && v >= 0 && v < mask.rows) {

                    // Check the foreground mask
                    if (mask.at<uchar>(imgY, imgX) > 0 && imageIndex == 0) {
                        // Carve out voxels in the first image
                        vol.set(x, y, z, -1.0);
                    } else if (mask.at<uchar>(imgY, imgX) > 0 && vol.get(x, y, z) != 0) {
                        // Keep carving out voxels if they are not already carved in subsequent images
                        vol.set(x, y, z, -1.0);
                    } else if (mask.at<uchar>(imgY, imgX) == 0 && vol.get(x, y, z) != 0 ) {
                        // Un-carve voxels if they are background (black) in subsequent images
                        vol.set(x, y, z, 0.0);
                    }        
                } else {
                    vol.set(x, y, z, 0);
                }
            }
        }
    }
    
}

// Save 3D points and camera positions to a PLY file
void saveToPLY(const std::string& filename,
               const std::vector<cv::Point3d>& markerPoints,
               const std::vector<cv::Point3d>& cameraPositions,
               const std::vector<cv::Vec3d>& cameraDirections) {
    std::ofstream plyFile(filename);

    if (!plyFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Calculate total number of points
    size_t numVertices = markerPoints.size() + cameraPositions.size();

    // Write PLY header
    plyFile << "ply\n";
    plyFile << "format ascii 1.0\n";
    plyFile << "element vertex " << numVertices << "\n";
    plyFile << "property float x\n";
    plyFile << "property float y\n";
    plyFile << "property float z\n";
    plyFile << "property uchar red\n";
    plyFile << "property uchar green\n";
    plyFile << "property uchar blue\n";
    plyFile << "end_header\n";

    // Write marker points
    for (const auto& point : markerPoints) {
        plyFile << point.x << " " << point.y << " " << point.z
                << " 0 255 0\n";  // Green for markers
    }

    // Write camera positions
    for (size_t i = 0; i < cameraPositions.size(); ++i) {
        const auto& position = cameraPositions[i];
        plyFile << position.x << " " << position.y << " " << position.z
                << " 255 0 0\n";  // Red for cameras
    }

    plyFile.close();
    std::cout << "Saved PLY file: " << filename << std::endl;
}

// Convert rotation and translation to camera position in world coordinates
cv::Point3d computeCameraPosition(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat camPos = -R.t() * t;
    return cv::Point3d(camPos.at<double>(0), camPos.at<double>(1), camPos.at<double>(2));
}

void generateAndAssignMarkers(const cv::Mat& image, int rows, int cols, double markerSize, double markerSpacing, std::unordered_map<int, cv::Point3f>& assignedMarkers) {
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    detectArucoMarkers(image, markerCorners, markerIds);

    cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
    cv::imwrite("detected_markers.jpg", image);

    if (markerIds.empty()) {
        std::cout << "No markers detected." << std::endl;
        return;
    }
 
    std::vector<cv::Point3f> markerMap;
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            float x = j * (markerSize + markerSpacing);
            float y = i * (markerSize + markerSpacing);
            markerMap.emplace_back(x, y, 0.0f);
        }
    }

    // print marker map
    for (int i = 0; i < markerMap.size(); ++i) {
        std::cout << "Marker " << i << ": " << markerMap[i] << std::endl;
    }

    for (size_t i = 0; i < markerIds.size(); ++i) {
        assignedMarkers[i] = markerMap[i];
    }

    // Print assigned markers
    for (const auto& [id, point] : assignedMarkers) {
        std::cout << "Marker " << id << ": " << point << std::endl;
    }
}

void performVoxelCarving(const std::vector<cv::Mat>& arucoImages, const std::vector<cv::Mat>& maskedImages, const std::unordered_map<int, cv::Point3f>& assignedMarkers, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const std::string& outputFilename) {
    const int gridSize = 100;
    const float voxelSize = 0.005f;
    Volume vol(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), gridSize, gridSize, gridSize, 1);
    // Initialize the volume with zeros
    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            for (int z = 0; z < gridSize; ++z) {
                vol.set(x, y, z, 0.0);
            }
        }
    }

    for (size_t i = 0; i < arucoImages.size(); ++i) {
        const cv::Mat& arucoImage = arucoImages[i];
        const cv::Mat& maskedImage = maskedImages[i];

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        detectArucoMarkers(arucoImage, markerCorners, markerIds);

        if (markerIds.empty()) {
            std::cout << "No markers detected in image " << i << "." << std::endl;
            continue;
        }

        // Identify a marker
        int idIndex = 0;
        int id = markerIds[0]; 

        float halfSize = 0.015f;
        std::vector<cv::Point3f> markerPoints = {
            cv::Point3f(assignedMarkers.at(id).x - halfSize, assignedMarkers.at(id).y - halfSize, 0.0f),
            cv::Point3f(assignedMarkers.at(id).x + halfSize, assignedMarkers.at(id).y - halfSize, 0.0f),
            cv::Point3f(assignedMarkers.at(id).x + halfSize, assignedMarkers.at(id).y + halfSize, 0.0f),
            cv::Point3f(assignedMarkers.at(id).x - halfSize, assignedMarkers.at(id).y + halfSize, 0.0f)
        };

        // print marker points
        std::cout << "Marker id: " << id << std::endl;
        for (int j = 0; j < markerPoints.size(); ++j) {
            std::cout << "Marker Points " << j << ": " << markerPoints[j] << std::endl;
        }
        // print marker corners
        for (int j = 0; j < markerCorners[idIndex].size(); ++j) {
            std::cout << "Marker Corner " << j << ": " << markerCorners[idIndex][j] << std::endl;
        }

        cv::Mat rvecs, tvecs;
        // cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.028, cameraMatrix, distCoeffs, rvecs, tvecs);
        bool found = cv::solvePnPRansac(markerPoints, markerCorners[idIndex], cameraMatrix, distCoeffs, rvecs, tvecs);

        // Debug marker pose estimates
        if (rvecs.empty() || tvecs.empty()) {
            std::cerr << "Error: Pose estimation failed for image " << i << ". Empty rvecs/tvecs." << std::endl;
            continue;
        }

        std::cout << "Rvecs Type: " << rvecs.type() << ", Size: " << rvecs.size() << std::endl;
        std::cout << "Tvecs Type: " << tvecs.type() << ", Size: " << tvecs.size() << std::endl;

        updateVolume(rvecs, tvecs, maskedImage, cameraMatrix, vol, gridSize, voxelSize, i);

        // Visualize the camera positions
        std::vector<cv::Point3d> cameraPositions;
        cv::Mat R;
        cv::Rodrigues(rvecs, R);
        cameraPositions.push_back(computeCameraPosition(R, tvecs));
        std::vector<cv::Point3d> markerPoints3d;
        for (const auto& pt : markerPoints) {
            markerPoints3d.emplace_back(pt.x, pt.y, pt.z);
        }
        saveToPLY("visualization_" + std::to_string(i) + ".ply", markerPoints3d, cameraPositions, {});
        }

    SimpleMesh mesh;
    std::cout << "Volume: " << vol.getDimX() << " " << vol.getDimY() << " " << vol.getDimZ() << " " << vol.getData() << std::endl;
    double minValue, maxValue;
    vol.computeMinMaxValues(minValue, maxValue);
    std::cout << "Volume Min: " << minValue << ", Max: " << maxValue << std::endl;

    // export and visualize volume points
    std::vector<cv::Point3d> volumePoints;
    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            for (int z = 0; z < gridSize; ++z) {
                if (vol.get(x, y, z) == -1.0) {
                    volumePoints.emplace_back((x - gridSize / 2) * voxelSize, 
                                              (y - gridSize / 2) * voxelSize, 
                                              (z - gridSize / 2) * voxelSize);
                }
            }
        }
    }
    saveToPLY("volume_points.ply", volumePoints, {}, {});

    marchingCubes(vol, 0.0f, mesh);

    std::cout << "Number of vertices: " << mesh.GetVertices().size() << std::endl;
    std::cout << "Number of faces: " << mesh.GetTriangles().size() << std::endl;

    mesh.WriteMesh(outputFilename);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <masked_image_directory> <calibration_results.yml> <output.off>" << std::endl;
        return -1;
    }

    std::string imageDirectory = argv[1];
    std::string maskedImageDirectory = argv[2];
    std::string calibrationFile = argv[3];
    std::string outputFilename = argv[4];

    cv::Mat cameraMatrix, distCoeffs;
    loadCalibrationResults(calibrationFile, cameraMatrix, distCoeffs);

    std::vector<cv::Mat> arucoImages;
    for ( auto& entry : fs::directory_iterator(imageDirectory)) {
        std::cout << "Reading image: " << entry.path().string() << std::endl;
        cv::Mat arucoImage = cv::imread(entry.path().string());
        if (!arucoImage.empty()) {
            arucoImages.push_back(arucoImage);
        }
    }
    if (arucoImages.empty()) {
        std::cerr << "No valid aruco images found in directory: " << imageDirectory << std::endl;
        return -1;
    }

    // Masked images
    std::vector<cv::Mat> maskedImages;
    for ( auto& entry : fs::directory_iterator(maskedImageDirectory)) {
        std::cout << "Reading masked image: " << entry.path().string() << std::endl;
        cv::Mat maskedImage = cv::imread(entry.path().string());
        if (!maskedImage.empty()) {
            maskedImages.push_back(maskedImage);
        }
    }
    if (maskedImages.empty()) {
        std::cerr << "No valid masked images found in directory: " << maskedImageDirectory << std::endl;
        return -1;
    }

    // Detect markers and assign them to the marker map
    cv::Mat mapImage = cv::imread("./markers.jpg");
    if (mapImage.empty()) {
        std::cerr << "Error: Could not read the marker map image." << std::endl;
        return -1;
    }
    std::unordered_map<int, cv::Point3f> assignedMarkers;
    generateAndAssignMarkers(mapImage, 5, 7, 0.03, 0.0075, assignedMarkers);
    
    performVoxelCarving(arucoImages, maskedImages, assignedMarkers, cameraMatrix, distCoeffs, outputFilename);

    return 0;
}