#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "ColorPool.h"
#include "Contour.h"
using namespace std;

const string ORIGINAL  = "ORIGINAL_IMAGE_WINDOW";
const string THRESHOLD = "THRESHOLD_IMAGE_WINDOW";

const string ORIG_IMAGE_PATH = "testImage6.png";

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

void GetThresholdedImage(cv::Mat& grayImg, cv::Mat& threshImg) {
	threshold(grayImg, threshImg, 235, 255, cv::ThresholdTypes::THRESH_BINARY);
}

void ReverseBlackWhiteThresholdedImage(cv::Mat& threshImg) {
	threshImg.forEach<OneChannelPixel>([](OneChannelPixel& p, const int* position)->void {
		if (p == 255) {
			p = 0;
		}
		else {
			p = 255;
		}
		});
}

void GetRelevantContours(cv::Mat& threshImg, vector<vector<cv::Point>>& contours) {
	vector<cv::Vec4i> hierarchy;
	findContours(threshImg, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_NONE);

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


	cv::Mat grayImg, threshImg;
	GetGrayImage(origImg, grayImg);
	GetThresholdedImage(grayImg, threshImg);
	ReverseBlackWhiteThresholdedImage(threshImg);

	cv::namedWindow(THRESHOLD, cv::WINDOW_AUTOSIZE);
	cv::imshow(THRESHOLD, threshImg);


	vector<vector<cv::Point>> contours;
	GetRelevantContours(threshImg, contours);


	sort(contours.begin(),
		 contours.end(), 
		 [](vector<cv::Point>& lhs, vector<cv::Point>& rhs) { 
			return lhs.size() > rhs.size(); 
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

	cv::Mat origImgCopy = origImg.clone();
	for (size_t i = 0; i < extracted_contours.size(); ++i) {
		cv::Scalar color = extracted_contours[i].color;
		int object_type_id = extracted_contours[i].object_type_id;

		drawContours(origImgCopy, contours, i, color, 2);
		cv::putText(origImgCopy, to_string(object_type_id), extracted_contours[i].contour_points[0], cv::FONT_HERSHEY_SIMPLEX, 1.0, color);
	}


	cv::namedWindow(ORIGINAL, cv::WINDOW_AUTOSIZE);
	imshow(ORIGINAL, origImgCopy);

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}