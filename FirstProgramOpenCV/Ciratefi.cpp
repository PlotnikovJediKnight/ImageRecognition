#include "Ciratefi.h"
using namespace std;
using namespace cv;

const double Ciratefi::scales[Ciratefi::N_scales] = { 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 };

const int Ciratefi::angles[Ciratefi::M_angles] = { 0, 10, 20, 30, 40, 50, 60, 70, 80, 
												   90, 100, 110, 120, 130, 140, 150, 
												   160, 170, 180, 190, 200, 210, 220,
												   230, 240, 250, 260, 270, 280, 290,
												   300, 310, 320, 330, 340, 350};


void Ciratefi::DoCiratefi(Mat& templateImageQ, Mat& searchImageA) {
	GetScaledTemplateImagesQ(templateImageQ);

	multiscaleRotationInvariantCQ.resize(N_scales);
	for (auto& it : multiscaleRotationInvariantCQ) it.resize(L_radii);

	CalculateRadiiArray(scaledTemplateImagesQ[N_scales - 1]);

	vector<Point> firstGradeCandidatePixels = Cifi(searchImageA, false);
	Rafi(firstGradeCandidatePixels, templateImageQ, searchImageA, true);
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
	int radiiIncrementStep = round(static_cast<double>(squareSideHalf) / (Ciratefi::L_radii - 1)) - 1;

	radii[0] = 0; radii[1] = radiiIncrementStep;

	for (size_t i = 2; i < Ciratefi::L_radii; ++i) {
		radii[i] = radii[i - 1] + radiiIncrementStep;
	}
}

vector<double> GetResizedTillNoNegativeElements(const vector<double>& cqi, vector<double>& caxy) {
	auto minusOneIgnore = find(cqi.begin(), cqi.end(), -1.0);
	if (minusOneIgnore == cqi.end()) return cqi;

	assert(minusOneIgnore >= cqi.begin());
	size_t relevantElements = minusOneIgnore - cqi.begin();

	vector<double> toReturn(cqi.begin(), cqi.end());
	toReturn.resize(relevantElements);
	caxy.resize(relevantElements);
	return toReturn;
}

vector<Point> Ciratefi::Cifi(Mat& searchImageA, bool drawFirstGradeCandidatePixels) {

	for (size_t i = 0; i < N_scales; ++i)
		CalculateMultiscaleRotationInvariantFeaturesCQ(i, scaledTemplateImagesQ[i]);

	Calculate3DImageCA(searchImageA);

	vector<Point> firstGradePoints;
	probableScales.clear();
	const double threshold1 = 0.95;
	for (int y = 0; y < searchImageA.rows; ++y) {
		for (int x = 0; x < searchImageA.cols; ++x) {
			double maxValue = 0.0; int maxScaleIndex = 0;
			for (int i = 0; i < N_scales; i++) {
				vector<double> caxy = caImage[y][x];
				vector<double> cqi = GetResizedTillNoNegativeElements(multiscaleRotationInvariantCQ[i], caxy);

				double result = abs(GetCorrelation(cqi, caxy));
				if (result > maxValue) { maxValue = result; maxScaleIndex = i; }
			}

			if (maxValue >= threshold1) {
				firstGradePoints.push_back({ x, y });
				probableScales.push_back(scales[maxScaleIndex]);
			}
		}
	}

	cout << "!!!!" << endl;
	cout << firstGradePoints.size() << endl;
	cout << "!!!!" << endl;

	if (drawFirstGradeCandidatePixels) {
		for (auto& it : firstGradePoints) {
			circle(searchImageA, it, 2, Scalar::all(255));
		}
	}

	return firstGradePoints;
}

int Ciratefi::GetMaximumRadius(Mat& templateImageQ) {
	assert(templateImageQ.rows == templateImageQ.cols);
	int maxRadius = templateImageQ.rows / 2 - 1;
	return maxRadius;
}

vector<double> GetShiftedVector(vector<double> vect, int step) {
	if (step == 0) return vect;
	step %= vect.size();

	rotate(vect.begin(), vect.begin() + step, vect.end());
	return vect;
}

vector<Point> Ciratefi::Rafi(vector<Point>& firstGradeCandidatePixels, Mat& templateImageQ, Mat& searchImageA, bool drawSecondGradeCandidatePixels) {
	int maxRadius = GetMaximumRadius(templateImageQ);

	Point centerPoint(templateImageQ.cols / 2, templateImageQ.rows / 2);
	radialSamplesRQ = RadialSampling(templateImageQ, centerPoint.x, centerPoint.y, maxRadius);

	{
		int i = 0;
		raImage.clear(); raImage.resize(firstGradeCandidatePixels.size());
		for (auto& pixel : firstGradeCandidatePixels) {
			raImage[i] = RadialSampling(searchImageA, pixel.x, pixel.y, probableScales[i] * maxRadius);
			i++;
		}
	}

	const double threshold2 = 0.9;
	vector<Point> secondGradePoints;
	probableAngles.clear();

	int pixelIndex = 0;
	for (auto& pixel : firstGradeCandidatePixels) {
		double maxValue = 0.0; int maxAngleIndex = 0;

		for (int j = 0; j < M_angles; j++) {
			vector<double>& raxy = raImage[pixelIndex];
			vector<double> shiftedrq = GetShiftedVector(radialSamplesRQ, j);
			
			double result = abs(GetCorrelation(raxy, shiftedrq));
			if (result > maxValue) { maxValue = result; maxAngleIndex = j; }
		}

		if (maxValue >= threshold2) {
			secondGradePoints.push_back({ pixel });
			probableScales.push_back(angles[maxAngleIndex]);
		}

		pixelIndex++;
	}

	cout << "~!!!!~" << endl;
	cout << secondGradePoints.size() << endl;
	cout << "~!!!!~" << endl;

	if (drawSecondGradeCandidatePixels) {
		for (auto& it : secondGradePoints) {
			circle(searchImageA, it, 2, Scalar::all(255));
		}
	}

	return secondGradePoints;
}

void Ciratefi::CalculateMultiscaleRotationInvariantFeaturesCQ(size_t scaleIndex, Mat& templateImageQ) {
	assert(templateImageQ.rows == templateImageQ.cols);
	int circleCenterCoord = templateImageQ.rows / 2;

	fill(
		multiscaleRotationInvariantCQ[scaleIndex].begin(),
		multiscaleRotationInvariantCQ[scaleIndex].end(),
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

void Ciratefi::ProcessPortion(int y1, int y2, Mat& searchImageA) {
	for (int y = y1; y <= y2; ++y) {
		for (int x = 0; x < searchImageA.cols; ++x) {
			caImage[y][x].reserve(L_radii);

			for (int k = 0; k < L_radii; ++k) {
				caImage[y][x].push_back(CircularSampling(searchImageA, x, y, radii[k]));
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
	unsigned int supportedThreads = thread::hardware_concurrency();
	assert(searchImageA.rows >= supportedThreads);

	int rowsPerThread = searchImageA.rows / supportedThreads;

	int fullDeploymentCount = searchImageA.rows / rowsPerThread;
	int leftOver = searchImageA.rows % rowsPerThread;

	int currRowLeft = 0;
	for (int i = 0; i < fullDeploymentCount; ++i) {
		futures.push_back(
			async(launch::async, &Ciratefi::ProcessPortion, ref(*this), currRowLeft, currRowLeft + rowsPerThread - 1, ref(searchImageA))
		);
		currRowLeft += rowsPerThread;
	}

	if (leftOver > 0) {
		futures.push_back(
			async(launch::async, &Ciratefi::ProcessPortion, ref(*this), currRowLeft, currRowLeft + leftOver - 1, ref(searchImageA))
		);
	}

	for (auto& it : futures) {
		it.wait();
	}
	cout << "Finished calculating CA" << endl;
}

double GetVectorMagnitude(const vector<double>& vect) {
	double squared_sum = 0.0;
	for (auto& it : vect) squared_sum += it * it;
	return squared_sum / vect.size();;
}

double Ciratefi::GetCorrelation(const vector<double>& x, const vector<double>& y) {
	const double tbeta = 0.1, tgamma = 1.0;

	vector<double> standX = GetStandardizedVector(x);
	vector<double> standY = GetStandardizedVector(y);

	double meanY = GetMean(y);
	double meanX = GetMean(x);

	double squaredStandX = inner_product(standX.begin(), standX.end(), standX.begin(), 0.0);

	const double beta = 
		  inner_product(standX.begin(), standX.end(), standY.begin(), 0.0)
		/ squaredStandX;

	const double gamma = meanY - beta * meanX;

	const double rXY = beta * squaredStandX / GetVectorMagnitude(standX) / GetVectorMagnitude(standY);

	const double absBeta = abs(beta);
	const double absGamma = abs(gamma);

	if (absBeta <= tbeta || 1 / tbeta <= absBeta || absGamma > tgamma) return 0;
	return rXY;
}

double Ciratefi::GetMean(const vector<double>& vect) {
	return accumulate(vect.begin(), vect.end(), 0.0) / vect.size();
}

vector<double> operator-(const vector<double>& lhs, double rhs) {
	vector<double> lhsCopy(lhs.begin(), lhs.end());
	for (auto& it : lhsCopy) it -= rhs;
	return lhsCopy;
}

vector<double> Ciratefi::GetStandardizedVector(const vector<double>& vect) {
	return vect - GetMean(vect);
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

vector<double>  Ciratefi::RadialSampling(cv::Mat& sampledImage, int x, int y, int r){
	Point centerPoint(x, y);
	Point firstPoint(x, y - r);
	vector<Point> circlePoints;
	ellipse2Poly(centerPoint, { r, r }, 0, 0, 360, 10, circlePoints);
	circlePoints.resize(circlePoints.size() - 1);

	int angleIndex = 0;
	vector<double> radialSamples; radialSamples.resize(M_angles);

	for (auto& currPoint : circlePoints) {
		LineIterator it(sampledImage, centerPoint, currPoint, 8);
		size_t pixelAlongLineSum = 0;

		for (int i = 0; i < it.count; i++) {
			pixelAlongLineSum += uchar(*it);
			it++;
		}

		radialSamples[angleIndex++] = pixelAlongLineSum / static_cast<double>(it.count);
	}

	return radialSamples;
}


