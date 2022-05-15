#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <cassert>
#include <vector>
#include <map>
#include <algorithm>
#include <future>
#include <iostream>

class Ciratefi {
public:
	void DoCiratefi(cv::Mat& templateImageQ, cv::Mat& searchImageA);
private:
	static const size_t N_scales = 6;
	static const double scales[];

	static const size_t L_radii = 13;

	int radii[L_radii];
	cv::Mat scaledTemplateImagesQ[N_scales];
	double multiscaleRotationInvariantCQ[N_scales][L_radii];
	std::vector<std::vector<std::vector<double>>> caImage;

	double CircularSampling(cv::Mat& sampledImage, int x, int y, int r);
	void Cifi(cv::Mat& searchImageA, bool drawFirstGradeCandidatePixels);
	void GetScaledTemplateImagesQ(cv::Mat& templateImageQ);
	void CalculateRadiiArray(cv::Mat& templateImageQ);
	void CalculateMultiscaleRotationInvariantFeaturesCQ(size_t scaleIndex, cv::Mat& templateImageQ);

	void ProcessPortion(int x1, int x2, cv::Mat& searchImageA);
	void Calculate3DImageCA(cv::Mat& searchImageA);
};
