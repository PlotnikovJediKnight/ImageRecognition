#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "ColorPool.h"
#include "Contour.h"

using namespace std;

const string ORIGINAL   = "ORIGINAL_IMAGE_WINDOW";
const string THRESHOLD  = "THRESHOLD_IMAGE_WINDOW";
const string SOBEL	    = "SOBEL_IMAGE_WINDOW";
const string PITCHBLACK = "PITCH_BLACK_WINDOW";

const string ORIG_IMAGE_PATH = "testImage3.jpg";

using OneChannelPixel = uchar;

const double MATCH_SHAPES_EPSILON_I3 = 0.1;

int GetNextObjectType() {
	static int object_type = 0;
	if (object_type == ColorPool::MAX_COLORS) throw runtime_error("Out of object ids!");
	return object_type++;
}

void GetGrayImage(cv::Mat& origImg, cv::Mat& grayImg) {
	cvtColor(origImg, grayImg, cv::ColorConversionCodes::COLOR_BGR2GRAY);
}

void GetBlurredImage(cv::Mat& origImg, cv::Mat& blurredImg) {
	cv::GaussianBlur(origImg, blurredImg, { 5, 5 }, 0);
}

void GetSobeledImage(cv::Mat& origGrayImg, cv::Mat& sobeledImg) {
	cv::Mat gradient_x, gradient_y;
	cv::Sobel(origGrayImg, gradient_x, CV_32F, 1, 0);
	cv::Sobel(origGrayImg, gradient_y, CV_32F, 0, 1);
	
	cv::Mat abs_gradient_x, abs_gradient_y;
	cv::convertScaleAbs(gradient_x, abs_gradient_x);
	cv::convertScaleAbs(gradient_y, abs_gradient_y);

	cv::addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0, sobeledImg);
}

void GetNoiseRemovedImage(cv::Mat& origImg, cv::Mat& noiseRemovedImg) {
	auto mean = cv::mean(origImg);
	cv::threshold(origImg, noiseRemovedImg, mean.val[0], 0, cv::THRESH_TOZERO);
}

void GetShadowsRemovedImage(cv::Mat& origImg, cv::Mat& shadowRemovedImg) {
	cv::Mat blankMask = cv::Mat::zeros(origImg.size(), CV_8UC3);
	cv::Mat original = origImg.clone();
	cv::Mat hsv;

	cv::cvtColor(origImg, hsv, cv::ColorConversionCodes::COLOR_BGR2HSV);
	cv::Mat lower{ 41, 57, 78 };
	cv::Mat upper{ 145, 255, 255};

	cv::Mat mask;
	cv::inRange(hsv, lower, upper, mask);


	vector<cv::Vec4i> hierarchy;
	vector<vector<cv::Point>> contours;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
	cv::drawContours(blankMask, contours, -1, cv::Scalar{0, 0, 255}, 2);
	cv::namedWindow("mask", cv::WINDOW_NORMAL);
	imshow("mask", blankMask);
	/*	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		cnts = sorted(cnts, key = cv::contourArea, reverse = True)
		for c in cnts :
	cv::drawContours(blank_mask, [c], -1, (255, 255, 255), -1)
		break

		result = cv::bitwise_and(original, blank_mask)
		*/
}

void GetRelevantContours(cv::Mat& origImg, vector<vector<cv::Point>>& contours) {
	vector<cv::Vec4i> hierarchy;
	findContours(origImg, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

	
	const size_t RELEVANT_SIZE = 20;
	contours.erase(
		remove_if(
			contours.begin(),
			contours.end(),
			[=](vector<cv::Point>& i_vector) {
				return i_vector.size() < RELEVANT_SIZE;
			}),
		contours.end()
	);

	
	const double RELEVANT_AREA = origImg.rows / 100.0 * origImg.cols * 1.1;
	contours.erase(
		remove_if(
			contours.begin(),
			contours.end(),
			[=](vector<cv::Point>& i_vector) {
				return cv::contourArea(i_vector) < RELEVANT_AREA;
			}),
		contours.end()
	);
	
}

void FillExtractedContours(vector<vector<cv::Point>>& contours, vector<Contour>& extracted_contours) {
	extracted_contours.reserve(contours.size());

	for (auto& contour : contours) {
		extracted_contours.emplace_back(contour);
	}
}

int main(int argc, char** argv) {

	cv::Mat origImg = cv::imread(ORIG_IMAGE_PATH, -1);
	if (origImg.empty()) return -1;


	cv::Mat blurredImg, blurredGrayImg, sobelImg, noiseRemovedImg, pitchBlackCanvas, cutOutContours, shadowsRemoved;

	GetBlurredImage(origImg, blurredImg);
	GetGrayImage(blurredImg, blurredGrayImg);
	GetSobeledImage(blurredGrayImg, sobelImg);
	GetNoiseRemovedImage(sobelImg, noiseRemovedImg);

	//cv::namedWindow(SOBEL, cv::WINDOW_NORMAL);
	//imshow(SOBEL, sobelImg);

	
	vector<vector<cv::Point>> contours;
	GetRelevantContours(noiseRemovedImg, contours);


	sort(contours.begin(),
		 contours.end(), 
		 [](vector<cv::Point>& lhs, vector<cv::Point>& rhs) { 
			return lhs.size() > rhs.size(); 
	});


	pitchBlackCanvas = cv::Mat::zeros(origImg.size(), CV_8UC3);
	cv::drawContours(pitchBlackCanvas, contours, -1, { 255, 255, 255, 255 }, -1);
	cv::bitwise_and(pitchBlackCanvas, origImg, cutOutContours);

	cv::Mat output = cutOutContours.clone();

	cv::namedWindow(PITCHBLACK, cv::WINDOW_NORMAL);
	cv::createTrackbar("HMin", PITCHBLACK, 0, 179);
	cv::createTrackbar("SMin", PITCHBLACK, 0, 255);
	cv::createTrackbar("VMin", PITCHBLACK, 0, 255);
	cv::createTrackbar("HMax", PITCHBLACK, 0, 179);
	cv::createTrackbar("SMax", PITCHBLACK, 0, 255);
	cv::createTrackbar("VMax", PITCHBLACK, 0, 255);

	cv::setTrackbarPos("HMax", PITCHBLACK, 179);
	cv::setTrackbarPos("SMax", PITCHBLACK, 255);
	cv::setTrackbarPos("VMax", PITCHBLACK, 255);

	int hMin = 0, sMin = 0, vMin = 0, hMax = 0, sMax = 0, vMax = 0,
		phMin = 0, psMin = 0, pvMin = 0, phMax = 0, psMax = 0, pvMax = 0;

	int waitTime = 33;

	while (true) {

		hMin = cv::getTrackbarPos("HMin", PITCHBLACK);
		sMin = cv::getTrackbarPos("SMin", PITCHBLACK);
		vMin = cv::getTrackbarPos("VMin", PITCHBLACK);

		hMax = cv::getTrackbarPos("HMax", PITCHBLACK);
		sMax = cv::getTrackbarPos("SMax", PITCHBLACK);
		vMax = cv::getTrackbarPos("VMax", PITCHBLACK);

		cv::Mat lower{ hMin, sMin, vMin };
		cv::Mat upper{ hMax, sMax, vMax };

		cv::Mat hsv, mask, maskThreeChannel;
		cv::cvtColor(cutOutContours, hsv, cv::ColorConversionCodes::COLOR_BGR2HSV);
		cv::inRange(hsv, lower, upper, mask);

		cv::cvtColor(mask, maskThreeChannel, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		output = cutOutContours & maskThreeChannel;

		if ((phMin != hMin) || (psMin != sMin) || (pvMin != vMin) || (phMax != hMax) || (psMax != sMax) || (pvMax != vMax)) {
			printf("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)\n", hMin, sMin, vMin, hMax, sMax, vMax);
				phMin = hMin;
				psMin = sMin;
				pvMin = vMin;
				phMax = hMax;
				psMax = sMax;
				pvMax = vMax;
		}

		imshow(PITCHBLACK, output);
		if (cv::waitKey(33) >= 0) break;
	}

	//imshow(PITCHBLACK, cutOutContours);

	/*
	GetShadowsRemovedImage(cutOutContours, shadowsRemoved);

	
	vector<Contour> extracted_contours;
	FillExtractedContours(contours, extracted_contours);

	for (auto& contour1 : extracted_contours) {
		if (contour1.object_type_id == -1) {
			int current_object_type_id = GetNextObjectType();
			cv::Scalar current_object_type_color = ColorPool::Instance().GetNextColor();
			for (auto& contour2 : extracted_contours) {
				if (contour2.object_type_id == -1) {
					if (cv::matchShapes(contour1.contour_points, contour2.contour_points, cv::CONTOURS_MATCH_I3, 0) < MATCH_SHAPES_EPSILON_I3) {
						contour2.color = current_object_type_color;
						contour2.object_type_id = current_object_type_id;
					}
				}
			}
		}
	}

	cv::Mat origImgCopy = origImg.clone();
	for (size_t i = 0; i < extracted_contours.size(); ++i) {
		cv::Scalar color = extracted_contours[i].color;
		int object_type_id = extracted_contours[i].object_type_id;

		drawContours(origImgCopy, contours, i, color, 2);
		cv::putText(origImgCopy, to_string(object_type_id), extracted_contours[i].contour_points[0], cv::FONT_HERSHEY_SIMPLEX, 1.0, color);
	}


	cv::namedWindow(ORIGINAL, cv::WINDOW_NORMAL);
	imshow(ORIGINAL, origImgCopy);
	*/

	//cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}