#include "../methods/parametersStereo.h"

using namespace cv;

bool check_image(const cv::Mat &image, std::string name)
{
	if (!image.data)
	{
		std::cerr << name << " data not loaded.\n";
		return false;
	}
	return true;
}


bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2)
{
	if (img1.cols != img2.cols || img1.rows != img2.rows)
	{
		std::cerr << "Images' dimensions do not corresponds.";
		return false;
	}
	return true;
}

/**
* @brief convert cv::Point2d(in OpenCV) to cv::Point(in OpenCV)
* @param point
* @return
*/
Point2d PointF2D(Point2f point)
{
	Point2d pointD = Point2d(point.x, point.y);
	return  pointD;
}
/**
 * \brief convert std::vector<cv::Point> to std::vector<cv::Point2d>
 * \param pts
 * \return
 */
std::vector<Point2d> VecPointF2D(std::vector<Point2f> pts)
{
	std::vector<Point2d> ptsD;
	for (std::vector<Point2f>::iterator iter = pts.begin(); iter != pts.end(); ++iter) {
		ptsD.push_back(PointF2D(*iter));
	}
	return ptsD;
}
