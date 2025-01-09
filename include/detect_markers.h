#ifndef DETECT_MARKERS_H
#define DETECT_MARKERS_H

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <vector>

void detectArucoMarkers(const cv::Mat& img, std::vector<std::vector<cv::Point2f>>& markerCorners, std::vector<int>& markerIds);

#endif // DETECT_MARKERS_H