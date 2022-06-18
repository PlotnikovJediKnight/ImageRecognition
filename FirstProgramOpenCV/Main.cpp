#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "ColorPool.h"
#include "Contour.h"

using namespace std;

const string ORIGINAL   = "ORIGINAL_IMAGE_WINDOW";
const string THRESHOLD  = "THRESHOLD_IMAGE_WINDOW";
const string PITCHBLACK = "PITCH_BLACK_WINDOW";

const string ORIG_IMAGE_PATH = "try3_12.jpg";

using OneChannelPixel = uchar;

const double MATCH_SHAPES_EPSILON_I3 = 0.1;

int GetNextObjectType(bool resetCounter) {
	static int object_type = 0;
	if (resetCounter) {
		object_type = -1;
	}
	if (object_type == ColorPool::MAX_COLORS) { throw runtime_error("Out of object ids!"); }
	return object_type++;
}

void GetGrayImage(cv::Mat& origImg, cv::Mat& grayImg) {
	cvtColor(origImg, grayImg, cv::ColorConversionCodes::COLOR_BGR2GRAY);
}

void GetThresholdImage(cv::Mat& origImg, cv::Mat& threshImg) {
	cv::adaptiveThreshold(origImg, threshImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::ThresholdTypes::THRESH_BINARY_INV, 11, 2);
}

void GetCannyImage(cv::Mat& origImg, cv::Mat& cannyImg) {
	cv::Canny(origImg, cannyImg, 1, 120, 3, true);
}

void GetNoiseRemovedImage(cv::Mat& origImg, cv::Mat& noiseRemovedImg) {
	auto mean = cv::mean(origImg);
	cv::threshold(origImg, noiseRemovedImg, mean.val[0], 0, cv::THRESH_TOZERO);
}

void GetRelevantContours(cv::Mat& origImg, vector<vector<cv::Point>>& contours, bool areaTakenIntoAccount) {
	vector<cv::Vec4i> hierarchy;
	findContours(origImg, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_TC89_L1);

	
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
		const double RELEVANT_AREA = origImg.rows / 100.0 * origImg.cols * 0.10;
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

cv::Rect GetCropBoundingRect(cv::Mat& rotatedImage) {
	size_t rows = rotatedImage.rows;
	size_t cols = rotatedImage.cols;

	const cv::Vec3b blackPixel( 0, 0, 0 );

	size_t y_min = rows - 1;
	for (size_t x = 0; x < cols; ++x) {
		for (size_t y = 0; y < rows; ++y) {
			if (rotatedImage.at<cv::Vec3b>(y, x) != blackPixel) {
				if (y < y_min) {
					y_min = y;
				}
				break;
			}
		}
	}

	size_t y_max = 0;
	for (size_t x = 0; x < cols; ++x) {
		for (size_t y = rows - 1; y > 0; --y) {
			if (rotatedImage.at<cv::Vec3b>(y, x) != blackPixel) {
				if (y > y_max) {
					y_max = y;
				}
				break;
			}
		}
	}

	size_t x_min = cols - 1;
	for (size_t y = 0; y < rows; ++y) {
		for (size_t x = 0; x < cols; ++x) {
			if (rotatedImage.at<cv::Vec3b>(y, x) != blackPixel) {
				if (x < x_min) {
					x_min = x;
				}
				break;
			}
		}
	}

	size_t x_max = 0;
	for (size_t y = 0; y < rows; ++y) {
		for (size_t x = cols - 1; x > 0; --x) {
			if (rotatedImage.at<cv::Vec3b>(y, x) != blackPixel) {
				if (x > x_max) {
					x_max = x;
				}
				break;
			}
		}
	}

	cv::Rect toReturn; toReturn.x = x_min; toReturn.y = y_min; toReturn.width = x_max - x_min; toReturn.height = y_max - y_min;
	return toReturn;
}



void DoTemplateMatching(cv::Mat& templatePatch, vector<Contour>& extractedContours, vector<cv::Mat> itemsOnPitchBlack, size_t patchIndex) {
	constexpr double MATCH_PASSED_VALUE = 0.65;
	constexpr size_t ANGLE_SAMPLES = 30;
	constexpr double angle = 360.0 / ANGLE_SAMPLES;

	vector<cv::Mat> templateAngleSamples; templateAngleSamples.reserve(ANGLE_SAMPLES);
	cv::Point2f center(templatePatch.cols * 0.5, templatePatch.rows * 0.5);

	for (size_t i = 0; i < ANGLE_SAMPLES + 1; i++) {
		double rotationAngle = angle * i;
		{
			cv::Mat rot_mat = cv::getRotationMatrix2D(center, rotationAngle, 1.0);
			cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), templatePatch.size(), rotationAngle).boundingRect2f();

			rot_mat.at<double>(0, 2) += bbox.width / 2.0 - templatePatch.cols / 2.0;
			rot_mat.at<double>(1, 2) += bbox.height / 2.0 - templatePatch.rows / 2.0;

			cv::Mat dst;
			cv::warpAffine(templatePatch, dst, rot_mat, bbox.size());
			
			cv::Mat final;
			cv::Rect roi = GetCropBoundingRect(dst);
			dst(roi).copyTo(final);

			templateAngleSamples.push_back(final);
		}
	}

	int current_group_id = GetNextObjectType(false);
	extractedContours[patchIndex].object_type_id = current_group_id;

	cout << patchIndex << ": ";

	for (auto& templateSample : templateAngleSamples) {
		for (size_t i = 0; i < itemsOnPitchBlack.size(); ++i) {

			if (i != patchIndex && extractedContours[i].object_type_id == -1) {

				cv::Mat result;
				cv::matchTemplate(itemsOnPitchBlack[i], templateSample, result, cv::TM_CCOEFF_NORMED);

				double maxValue = -1.0;
				cv::Point maxPoint;
				cv::minMaxLoc(result, NULL, &maxValue, NULL, &maxPoint);

				if (maxValue >= MATCH_PASSED_VALUE) {
					extractedContours[i].object_type_id = current_group_id;
					cout << i << " (" << maxValue << ") ";
				}

			}

		}
	}

	cout << endl;
}

int main(int argc, char** argv) {

	cv::Mat origImg = cv::imread(ORIG_IMAGE_PATH, -1);
	if (origImg.empty()) return -1;


	cv::Mat blurredImg, noiseRemovedImg, sobelImg, 
			blurredGrayImg, pitchBlackCanvas, cutOutContours;

	GetBlurredImage(origImg, blurredImg);
	GetGrayImage(blurredImg, blurredGrayImg);
	GetSobeledImage(blurredGrayImg, sobelImg);
	GetNoiseRemovedImage(sobelImg, noiseRemovedImg);
	
	vector<vector<cv::Point>> contours;
	GetRelevantContours(noiseRemovedImg, contours, true);

	pitchBlackCanvas = cv::Mat::zeros(origImg.size(), CV_8UC3);
	cv::drawContours(pitchBlackCanvas, contours, -1, { 255, 255, 255, 255 }, -1);
	cv::bitwise_and(pitchBlackCanvas, origImg, cutOutContours);


	vector<Contour> extracted_contours;
	FillExtractedContours(contours, extracted_contours);

	vector<cv::Mat> isolatedItemsMats;
	vector<cv::Mat> itemsOnPitchBlack;

	isolatedItemsMats.reserve(contours.size());
	itemsOnPitchBlack.reserve(contours.size());


	cv::Mat origImgCopy = origImg.clone();
	for (size_t i = 0; i < extracted_contours.size(); ++i) {
		cv::Scalar color = ColorPool::Instance().GetNextColor();
		int object_type_id = GetNextObjectType(false);

		cv::Mat cutOutItem;
		cv::Mat pitchBlackCanvas = cv::Mat::zeros(origImg.size(), CV_8UC3);
		cv::drawContours(pitchBlackCanvas, contours, i, { 255, 255, 255, 255 }, -1);
		cv::bitwise_and(pitchBlackCanvas, origImg, cutOutItem);

		{
			cv::Mat final;
			cv::Rect roi = boundingRect(contours[i]);
			cutOutItem(roi).copyTo(final);

			isolatedItemsMats.push_back(final);
			itemsOnPitchBlack.push_back(cutOutItem);
		}

		drawContours(origImgCopy, contours, i, color, 3);
		cv::putText(origImgCopy, to_string(object_type_id), extracted_contours[i].contour_points[0], cv::FONT_HERSHEY_COMPLEX, 5.0, color, 10);
	}
	
	cv::namedWindow("test1", cv::WINDOW_KEEPRATIO);
	cv::imshow("test1", cutOutContours);

	cv::namedWindow("test2", cv::WINDOW_KEEPRATIO);
	cv::imshow("test2", origImgCopy);

	GetNextObjectType(true);
	size_t itemIndex = 0;
	for (auto& isolatedItemMatrix : isolatedItemsMats) {
		if (extracted_contours[itemIndex].object_type_id == -1) {
			DoTemplateMatching(isolatedItemMatrix, extracted_contours, itemsOnPitchBlack, itemIndex);
		}
		itemIndex++;
	}
	
	itemIndex = 0;
	for (auto& contour : extracted_contours) {
		cv::drawContours(origImg, contours, itemIndex, ColorPool::Instance().GetColorById(contour.object_type_id), 3);
		itemIndex++;
	}

	cv::namedWindow("test3", cv::WINDOW_KEEPRATIO);
	cv::imshow("test3", origImg);

	cv::waitKey(0);
	cv::destroyAllWindows();

	cv::imwrite("initialRecognition.jpg", origImgCopy);
	cv::imwrite("finishedRecognition.jpg", origImg);

	return 0;
}