#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

using ContourColor = cv::Scalar;

class ColorPool {
public:
	static const size_t MAX_COLORS;
	static ColorPool& Instance();

	ColorPool(const ColorPool&) = delete;
	ColorPool& operator=(const ColorPool&) = delete;
	ContourColor GetNextColor();

private:
	ColorPool();
	~ColorPool();

	std::vector<ContourColor> stock;
};