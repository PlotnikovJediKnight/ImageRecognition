#include "Contour.h"

Contour::Contour(std::vector<cv::Point>& ref) : 
	object_type_id(-1),
	contour_points(ref),
	color(255, 255, 255) { }
