#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>
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

	static const size_t M_angles = 36;
	static const int angles[];

	cv::Mat scaledTemplateImagesQ[N_scales];
	std::vector<std::vector<double>> multiscaleRotationInvariantCQ;
	std::vector<std::vector<std::vector<double>>> caImage;
	std::vector<double> probableScales;

	std::vector<double> radialSamplesRQ;
	std::vector<std::vector<double>> raImage;
	std::vector<int> probableAngles;

	double CircularSampling(cv::Mat& sampledImage, int x, int y, int r);
	std::vector<double> RadialSampling(cv::Mat& sampledImage, int x, int y, int r);
	
	std::vector<cv::Point> Cifi(cv::Mat& searchImageA, bool drawFirstGradeCandidatePixels);
	std::vector<cv::Point> Rafi(std::vector<cv::Point>& firstGradeCandidatePixels, cv::Mat& templateImageQ, cv::Mat& searchImageA, bool drawSecondGradeCandidatePixels);

	void GetScaledTemplateImagesQ(cv::Mat& templateImageQ);
	void CalculateRadiiArray(cv::Mat& templateImageQ);
	void CalculateMultiscaleRotationInvariantFeaturesCQ(size_t scaleIndex, cv::Mat& templateImageQ);

	void ProcessPortion(int x1, int x2, cv::Mat& searchImageA);
	void Calculate3DImageCA(cv::Mat& searchImageA);

	double GetCorrelation(const std::vector<double>& x, const std::vector<double>& y);
	double GetMean(const std::vector<double>& vect);
	std::vector<double> GetStandardizedVector(const std::vector<double>& vect);

	int GetMaximumRadius(cv::Mat& templateImageQ);
};
