#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {
    // Parameters for Charuco board
    Size boardSize(8, 12); // 8x12 Charuco board
    float squareLength = 20.0f;  // Length of a square in mm
    float markerLength = 15.0f;  // Length of a marker in mm

    // Create Ptr for Dictionary
    Ptr<aruco::Dictionary> dictionary = makePtr<aruco::Dictionary>(aruco::getPredefinedDictionary(aruco::DICT_4X4_50));

    // Create Charuco board
    Ptr<aruco::CharucoBoard> board = makePtr<aruco::CharucoBoard>(boardSize, squareLength, markerLength, *dictionary);

    // Create DetectorParameters object
    Ptr<aruco::DetectorParameters> detectorParams = makePtr<aruco::DetectorParameters>();    
    Ptr<aruco::CharucoParameters> charucoParams = makePtr<aruco::CharucoParameters>();


    // Initialize CharucoDetector
    Ptr<aruco::CharucoDetector> detector = makePtr<aruco::CharucoDetector>(*board, *charucoParams, *detectorParams);

    // Folder containing the images of the Charuco board
    string folderPath = "../charucoBoard";  // Adjust path as needed

    // Check if the folder exists
    if (!fs::exists(folderPath)) {
        cerr << "The folder " << folderPath << " does not exist!" << endl;
        return -1;
    }

    vector<string> imagePaths;

    // Load all image file paths from the folder
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            imagePaths.push_back(entry.path().string());
        }
    }

    if (imagePaths.empty()) {
        cerr << "No images found in the folder: " << folderPath << endl;
        return -1;
    }

    // Collect data from each image
    vector<vector<Point2f>> allImagePoints;
    vector<vector<Point3f>> allObjectPoints; 
    Size imageSize;

    for (const auto& imagePath : imagePaths) {
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Could not load image: " << imagePath << endl;
            continue;
        }

        // Detect ChArUco board using detectBoard
        Mat charucoCorners, charucoIds;
        detector->detectBoard(image, charucoCorners, charucoIds);

        if (charucoCorners.total() > 3) {
            vector<Point3f> objectPoints;
            vector<Point2f> imagePoints;

            board->matchImagePoints(charucoCorners, charucoIds, objectPoints, imagePoints);

            if (imagePoints.empty() || objectPoints.empty()) {
                continue;
            }

            cout << "Valid points detected in image: " << imagePath << endl;

            // Store data
            allObjectPoints.push_back(objectPoints);
            allImagePoints.push_back(imagePoints);

            imageSize = image.size();
        } else {
            cerr << "Insufficient corners detected in image: " << imagePath << endl;
        }
    }

    if (allImagePoints.empty() || allObjectPoints.empty()) {
        cerr << "No valid data collected for calibration." << endl;
        return -1;
    }

    // Camera calibration
    Mat cameraMatrix, distCoeffs; 
    int calibrationFlags = 0;  // Define calibration flags, e.g., CALIB_ZERO_TANGENT_DIST
    double repError = calibrateCamera(allObjectPoints, allImagePoints, imageSize, cameraMatrix, distCoeffs,
                                      noArray(), noArray(), noArray(), noArray(), noArray(), calibrationFlags);

    cout << "Calibration successful!" << endl;
    cout << "Reprojection error: " << repError << endl;
    cout << "Camera Matrix: " << endl << cameraMatrix << endl;
    cout << "Distortion Coefficients: " << endl << distCoeffs << endl;

    // Save calibration data
    FileStorage fs("calibration.yml", FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "dist_coeffs" << distCoeffs;
    fs.release();

    cout << "Calibration data saved to calibration.yml" << endl;

    return 0;
}
