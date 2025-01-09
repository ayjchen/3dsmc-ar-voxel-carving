#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {
    // Paths and parameters
    string imageFolder = "../data_raw"; 
    Size boardSize(5, 7); // 5x7 gridboard
    float markerLength = 28.5f; // Marker size in mm
    float markerSeparation = 7.0f; // Marker separation in mm
    Ptr<aruco::Dictionary> dictionary = makePtr<aruco::Dictionary>(aruco::getPredefinedDictionary(aruco::DICT_5X5_50));

    // Create Charuco board and detector
    Ptr<aruco::GridBoard> gridBoard = makePtr<aruco::GridBoard>(boardSize, markerLength, markerSeparation, *dictionary);
    Ptr<aruco::DetectorParameters> detectorParams = makePtr<aruco::DetectorParameters>();    
    
    aruco::ArucoDetector detector(*dictionary, *detectorParams); 

    // Storage for calibration
    vector<vector<vector<Point2f>>> allMarkerCorners;
    vector<vector<int>> allMarkerIds;
    Size imageSize;

    // Process images in folder
    for (const auto& entry : fs::directory_iterator(imageFolder)) {
        string filePath = entry.path().string();
        Mat image = imread(filePath);

        if (image.empty()) {
            cerr << "Could not open image: " << filePath << endl;
            continue;
        }

        // Detect Markers
        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners, rejectedMarkers;
        detector.detectMarkers(image, markerCorners, markerIds, rejectedMarkers);

        allMarkerCorners.push_back(markerCorners);
        allMarkerIds.push_back(markerIds);
        imageSize = image.size();

        // Draw detected markers
        aruco::drawDetectedMarkers(image, markerCorners, markerIds);

        // Display the image
        imshow("Detected Board", image);
        waitKey(0); // Wait for key press to proceed to the next image
    }

    Mat cameraMatrix, distCoeffs;
    int calibrationFlags = 0; // You can set flags for specific calibration needs
    if(calibrationFlags & CALIB_FIX_ASPECT_RATIO) {
        cameraMatrix = Mat::eye(3, 3, CV_64F);
        double aspectRatio = 1.33; 
        cameraMatrix.at<double>(0, 0) = aspectRatio;
    }
 
    // Prepare data for calibration
    vector<Point3f> objectPoints;
    vector<Point2f> imagePoints;
    vector<Mat> processedObjectPoints, processedImagePoints;
    size_t nFrames = allMarkerCorners.size();
 
    for(size_t frame = 0; frame < nFrames; frame++) {
        Mat currentImgPoints, currentObjPoints;
 
        gridBoard->matchImagePoints(allMarkerCorners[frame], allMarkerIds[frame], currentObjPoints, currentImgPoints);
 
        if(currentImgPoints.total() > 0 && currentObjPoints.total() > 0) {
            processedImagePoints.push_back(currentImgPoints);
            processedObjectPoints.push_back(currentObjPoints);
        }
    }
 
    // Calibrate camera
    double repError = calibrateCamera(processedObjectPoints, processedImagePoints, imageSize, cameraMatrix, distCoeffs,
                                      noArray(), noArray(), noArray(), noArray(), noArray(), calibrationFlags);

    // Output results
    cout << "Calibration completed." << endl;
    cout << "Reprojection error: " << repError << endl;
    cout << "Camera matrix: \n" << cameraMatrix << endl;
    cout << "Distortion coefficients: \n" << distCoeffs << endl;

    // Save results
    FileStorage fs("calibration_results.yml", FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "dist_coeffs" << distCoeffs;
    fs.release();

    return 0;
}