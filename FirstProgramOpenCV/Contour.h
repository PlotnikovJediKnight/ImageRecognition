#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Contour {

	Contour(std::vector<cv::Point>&);

	int object_type_id;
	std::vector<cv::Point>& contour_points;
	cv::Scalar color;

};
