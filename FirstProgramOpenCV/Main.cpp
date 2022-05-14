#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "ColorPool.h"
#include "Contour.h"

#define HSV_GUI

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
	cv::Mat lower{ 0, 67, 0 };
	cv::Mat upper{ 179, 255, 255};

	cv::Mat mask;
	cv::inRange(hsv, lower, upper, mask);


	vector<cv::Vec4i> hierarchy;
	vector<vector<cv::Point>> contours;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
	cv::drawContours(blankMask, contours, -1, cv::Scalar{255, 255, 255}, -1);

	cv::bitwise_and(original, blankMask, shadowRemovedImg);
}

void GetRelevantContours(cv::Mat& origImg, vector<vector<cv::Point>>& contours, bool areaTakenIntoAccount) {
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

	if (areaTakenIntoAccount) {
		const double RELEVANT_AREA = origImg.rows / 100.0 * origImg.cols * 0.80;
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
	
}

void FillExtractedContours(vector<vector<cv::Point>>& contours, vector<Contour>& extracted_contours) {
	extracted_contours.reserve(contours.size());

	for (auto& contour : contours) {
		extracted_contours.emplace_back(contour);
	}
}

/*
int GetOuterCircleRadius(const cv::Rect& roi) {
	double a = roi.width / 2.0;
	double b = roi.height / 2.0;

	return static_cast<int>(ceil(sqrt(a * a + b * b)));
}
*/

int main(int argc, char** argv) {

	cv::Mat origImg = cv::imread(ORIG_IMAGE_PATH, -1);
	if (origImg.empty()) return -1;


	cv::Mat blurredImg, blurredGrayImg, sobelImg, noiseRemovedImg, 
			pitchBlackCanvas, cutOutContours, shadowsRemoved, shadowsRemovedGray;

	GetBlurredImage(origImg, blurredImg);
	GetGrayImage(blurredImg, blurredGrayImg);
	GetSobeledImage(blurredGrayImg, sobelImg);
	GetNoiseRemovedImage(sobelImg, noiseRemovedImg);

	
	vector<vector<cv::Point>> contours;
	GetRelevantContours(noiseRemovedImg, contours, true);

	pitchBlackCanvas = cv::Mat::zeros(origImg.size(), CV_8UC3);
	cv::drawContours(pitchBlackCanvas, contours, -1, { 255, 255, 255, 255 }, -1);
	cv::bitwise_and(pitchBlackCanvas, origImg, cutOutContours);

	#undef HSV_GUI
	#ifdef HSV_GUI
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

	#endif

	
	GetShadowsRemovedImage(cutOutContours, shadowsRemoved);
	GetGrayImage(shadowsRemoved, shadowsRemovedGray);

	contours.clear();
	GetRelevantContours(shadowsRemovedGray, contours, false);

	sort(contours.begin(),
		contours.end(),
		[](vector<cv::Point>& lhs, vector<cv::Point>& rhs) {
			return lhs.size() < rhs.size();
		});

	
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


	//!!!!//
	//!
	vector<cv::Mat> isolatedGrayscaleItemsMats;
	isolatedGrayscaleItemsMats.reserve(contours.size());
	//! 
	//!!!!//

	cv::Mat origImgCopy = origImg.clone();
	for (size_t i = 0; i < extracted_contours.size(); ++i) {
		cv::Scalar color = extracted_contours[i].color;
		int object_type_id = extracted_contours[i].object_type_id;

		drawContours(origImgCopy, contours, i, color, 2);
		{
			cv::Rect roi = boundingRect(contours[i]);

			float outerCircleRadius = 0.0;
			cv::Point2f circleCenter{ 0.0, 0.0 };
			cv::minEnclosingCircle(contours[i], circleCenter, outerCircleRadius);

			const float a = roi.width  / 2.0;
			const float b = roi.height / 2.0;

			int squareSide = static_cast<int>(outerCircleRadius * 2) + 1;
			int offsetRows = static_cast<int>(squareSide / 2.0 - b);
			int offsetColumns = static_cast<int>(squareSide / 2.0 - a);

			cv::Mat mask = cv::Mat::zeros(roi.size(), shadowsRemovedGray.type());
			drawContours(mask, contours, i, cv::Scalar::all(255), -1, 8, cv::noArray(), -1,	-roi.tl());

			cv::Mat semiFinal, final;
			shadowsRemovedGray(roi).copyTo(semiFinal, mask);

			cv::Mat temp(squareSide, squareSide, shadowsRemovedGray.type(), cv::Scalar::all(0));
			cv::Mat roiMat(temp(cv::Rect(offsetColumns, offsetRows, roi.width, roi.height)));
			semiFinal.copyTo(roiMat);
			final = temp.clone();

			isolatedGrayscaleItemsMats.push_back(final);
		}
		cv::putText(origImgCopy, to_string(object_type_id), extracted_contours[i].contour_points[0], cv::FONT_HERSHEY_SIMPLEX, 1.0, color);
	}
	
	size_t i = 0;
	for (auto& it : isolatedGrayscaleItemsMats) {
		cv::namedWindow(to_string(i), cv::WINDOW_KEEPRATIO);
		cv::imshow(to_string(i), it);
		i++;
	}

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}