#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

void detectArucoMarkers(const string& imagePath, const string& outputPath) {
    // Read the segmented image
    Mat img = imread(imagePath);
    if (img.empty()) {
        cerr << "Failed to load image " << imagePath << endl;
        return;
    }

    // Detector parameters and dictionary
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_5X5_50);

    // Detect ArUco markers
    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners;
    aruco::detectMarkers(img, dictionary, markerCorners, markerIds, detectorParams);

    if (markerIds.empty()) {
        cerr << "No ArUco markers detected in image " << imagePath << endl;
        return;
    }

    // Draw detected markers
    aruco::drawDetectedMarkers(img, markerCorners, markerIds);

    // Save the result
    imwrite(outputPath, img);
}

void processDataset(const string& inputDir, const string& outputDir) {
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" || entry.path().extension() == ".jpeg") {
            string inputPath = entry.path().string();
            string outputPath = (fs::path(outputDir) / entry.path().filename()).string();
            detectArucoMarkers(inputPath, outputPath);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_directory> <output_directory>" << endl;
        return -1;
    }

    string inputDirectory = argv[1];
    string outputDirectory = argv[2];

    processDataset(inputDirectory, outputDirectory);

    return 0;
}