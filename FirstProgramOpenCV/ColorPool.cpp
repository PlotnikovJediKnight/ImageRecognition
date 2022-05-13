#include "ColorPool.h"
using namespace std;
using namespace cv;

const size_t ColorPool::MAX_COLORS = 100;

ColorPool& ColorPool::Instance(){
	static ColorPool _instance;
	return _instance;
}

ContourColor ColorPool::GetNextColor(){
	static size_t ind = 0;
	if (ind == MAX_COLORS) { throw runtime_error("Out of colors!"); }
	return stock[ind++];
}

ColorPool::ColorPool() {
	stock = {
		Scalar(255, 0, 0),
		Scalar(0, 255, 0),
		Scalar(0, 0, 255),
		Scalar(255, 255, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255),
		Scalar(255, 255, 220),
		Scalar(203, 255, 192),
		Scalar(32, 83, 75),
		Scalar(32, 114, 243),
		Scalar(153, 51, 51),
		Scalar(255, 153, 153),
		Scalar(128, 0, 128),
		Scalar(153, 204, 255),
		Scalar(204, 255, 204),
		Scalar(160, 208, 255),
		Scalar(128, 128, 128),
		Scalar(128, 0, 128),
		Scalar(0, 84, 211),
		Scalar(94, 73, 52),
		Scalar(255, 0, 0),
		Scalar(0, 255, 0),
		Scalar(0, 0, 255),
		Scalar(255, 255, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255),
		Scalar(255, 255, 220),
		Scalar(203, 255, 192),
		Scalar(32, 83, 75),
		Scalar(32, 114, 243),
		Scalar(153, 51, 51),
		Scalar(255, 153, 153),
		Scalar(128, 0, 128),
		Scalar(153, 204, 255),
		Scalar(204, 255, 204),
		Scalar(160, 208, 255),
		Scalar(128, 128, 128),
		Scalar(128, 0, 128),
		Scalar(0, 84, 211),
		Scalar(94, 73, 52),
		Scalar(255, 0, 0),
		Scalar(0, 255, 0),
		Scalar(0, 0, 255),
		Scalar(255, 255, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255),
		Scalar(255, 255, 220),
		Scalar(203, 255, 192),
		Scalar(32, 83, 75),
		Scalar(32, 114, 243),
		Scalar(153, 51, 51),
		Scalar(255, 153, 153),
		Scalar(128, 0, 128),
		Scalar(153, 204, 255),
		Scalar(204, 255, 204),
		Scalar(160, 208, 255),
		Scalar(128, 128, 128),
		Scalar(128, 0, 128),
		Scalar(0, 84, 211),
		Scalar(94, 73, 52),
		Scalar(255, 0, 0),
		Scalar(0, 255, 0),
		Scalar(0, 0, 255),
		Scalar(255, 255, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255),
		Scalar(255, 255, 220),
		Scalar(203, 255, 192),
		Scalar(32, 83, 75),
		Scalar(32, 114, 243),
		Scalar(153, 51, 51),
		Scalar(255, 153, 153),
		Scalar(128, 0, 128),
		Scalar(153, 204, 255),
		Scalar(204, 255, 204),
		Scalar(160, 208, 255),
		Scalar(128, 128, 128),
		Scalar(128, 0, 128),
		Scalar(0, 84, 211),
		Scalar(94, 73, 52),
		Scalar(255, 0, 0),
		Scalar(0, 255, 0),
		Scalar(0, 0, 255),
		Scalar(255, 255, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255),
		Scalar(255, 255, 220),
		Scalar(203, 255, 192),
		Scalar(32, 83, 75),
		Scalar(32, 114, 243),
		Scalar(153, 51, 51),
		Scalar(255, 153, 153),
		Scalar(128, 0, 128),
		Scalar(153, 204, 255),
		Scalar(204, 255, 204),
		Scalar(160, 208, 255),
		Scalar(128, 128, 128),
		Scalar(128, 0, 128),
		Scalar(0, 84, 211),
		Scalar(94, 73, 52),
	};
}

ColorPool::~ColorPool(){

}
