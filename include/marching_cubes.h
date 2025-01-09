#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include <vector>
#include <opencv2/opencv.hpp>

void marchingCubes(const cv::Mat& scalarField, float isoLevel, std::vector<cv::Point3f>& vertices, std::vector<std::vector<int>>& faces);

#endif // MARCHING_CUBES_H