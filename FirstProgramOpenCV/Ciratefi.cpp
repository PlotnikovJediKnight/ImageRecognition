#include "Ciratefi.h"
using namespace std;
using namespace cv;

const double Ciratefi::scales[Ciratefi::N_scales] = { 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 };

void Ciratefi::DoCiratefi(Mat& templateImageQ, Mat& searchImageA) {
	GetScaledTemplateImagesQ(templateImageQ);
	CalculateRadiiArray(scaledTemplateImagesQ[N_scales - 1]);

	Cifi(searchImageA, true);
}

void Ciratefi::GetScaledTemplateImagesQ(Mat& templateImageQ) {
	for (size_t i = 0; i <= 3; ++i) {
		resize(templateImageQ, scaledTemplateImagesQ[i], Size(), scales[i], scales[i], INTER_AREA);
	}

	scaledTemplateImagesQ[4] = templateImageQ.clone();
	resize(templateImageQ, scaledTemplateImagesQ[5], Size(), scales[5], scales[5], INTER_CUBIC);
}

void Ciratefi::CalculateRadiiArray(Mat& templateImageQ) {
	int squareSideHalf = templateImageQ.cols / 2;
	int radiiIncrementStep = squareSideHalf / (Ciratefi::L_radii - 1);

	radii[0] = 0; radii[1] = radiiIncrementStep;

	for (size_t i = 2; i < Ciratefi::L_radii; ++i) {
		radii[i] = radii[i - 1] + radiiIncrementStep;
	}
}

void Ciratefi::Cifi(Mat& searchImageA, bool drawFirstGradeCandidatePixels) {

	for (size_t i = 0; i < N_scales; ++i)
		CalculateMultiscaleRotationInvariantFeaturesCQ(i, scaledTemplateImagesQ[i]);

	Calculate3DImageCA(searchImageA);
}

void Ciratefi::CalculateMultiscaleRotationInvariantFeaturesCQ(size_t scaleIndex, Mat& templateImageQ) {
	assert(templateImageQ.rows == templateImageQ.cols);
	int circleCenterCoord = templateImageQ.rows / 2;

	fill(
		multiscaleRotationInvariantCQ[scaleIndex],
		static_cast<double*>(multiscaleRotationInvariantCQ[scaleIndex]) + L_radii,
		-1.0
	);

	for (size_t radiusIndex = 0; radiusIndex < L_radii; radiusIndex++) {
		int circleRadius = radii[radiusIndex];

		if (circleRadius > circleCenterCoord) {
			break;
		}

		if (circleRadius == circleCenterCoord) {
			circleRadius--;
		}

		multiscaleRotationInvariantCQ[scaleIndex][radiusIndex]
			= CircularSampling(templateImageQ, circleCenterCoord, circleCenterCoord, circleRadius);
	}
}

void Ciratefi::ProcessPortion(int x1, int x2, cv::Mat& searchImageA) {
	for (int x = x1; x <= x2; ++x) {
		for (int y = 0; y < searchImageA.cols; ++y) {
			caImage[x][y].reserve(L_radii);

			for (int k = 0; k < L_radii; ++k) {
				caImage[x][y].push_back(CircularSampling(searchImageA, x, y, radii[k]));
			}
		}
	}
}

void Ciratefi::Calculate3DImageCA(Mat& searchImageA){
	caImage.resize(searchImageA.rows);
	for (auto& it : caImage) {
		it.resize(searchImageA.cols);
	}

	vector<future<void>> futures;
	unsigned int supportedThreads = std::thread::hardware_concurrency();
	assert(searchImageA.rows >= supportedThreads);

	int rowsPerThread = searchImageA.rows / supportedThreads;

	int fullDeploymentCount = searchImageA.rows / rowsPerThread;
	int leftOver = searchImageA.rows % rowsPerThread;

	int currRowLeft = 0;
	for (int i = 0; i < fullDeploymentCount; ++i) {
		futures.push_back(
			async(launch::async, &Ciratefi::ProcessPortion, ref(*this), currRowLeft, currRowLeft + rowsPerThread - 1, ref(searchImageA))
		);
		currRowLeft += fullDeploymentCount;
	}

	if (leftOver > 0) {
		futures.push_back(
			async(launch::async, &Ciratefi::ProcessPortion, ref(*this), currRowLeft + 1, currRowLeft + leftOver - 1, ref(searchImageA))
		);
	}

	for (auto& it : futures) {
		it.wait();
	}
	cout << "Finished calculating CA" << endl;
}

double Ciratefi::CircularSampling(Mat& sampledImage, int x, int y, int r) {
	Point circleCenter(x, y);
	Size axes(r, r);
	vector<Point> pointsOnCircle;
	ellipse2Poly(circleCenter, axes, 0, 0, 360, 1, pointsOnCircle);

	

	size_t totalGrayscaleSum = 0;
	for (auto& circlePoint : pointsOnCircle) {
		if (circlePoint.x < 0 || circlePoint.y < 0 || 
			circlePoint.x >= sampledImage.cols || circlePoint.y >= sampledImage.rows)
			totalGrayscaleSum += 255;
		else
			totalGrayscaleSum += sampledImage.at<uchar>(circlePoint);
	}

	double averageGrayscale = static_cast<double>(totalGrayscaleSum) / pointsOnCircle.size();
	return averageGrayscale;
}


