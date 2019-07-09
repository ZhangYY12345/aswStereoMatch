#pragma once
#include <opencv2/opencv.hpp>
#include "parametersStereo.h"

struct MY_COMP_Point2i {
	bool operator()(const cv::Point& left, const cv::Point& right) const
	{
		if (left.x < right.x)
		{
			return true;
		}
		if (left.x == right.x && left.y < right.y)
		{
			return true;
		}

		return false;
	}
};

struct MY_COMP_Point3i {
	bool operator()(const cv::Point3i& left, const cv::Point3i& right) const
	{
		if (left.x < right.x)
		{
			return true;
		}
		if (left.x == right.x && left.y < right.y)
		{
			return true;
		}
		if (left.x == right.x && left.y == right.y && left.z == right.z)
		{
			return true;
		}

		return false;
	}
};

struct MY_COMP_vec3i
{
	bool operator()(const cv::Vec3i& left, const cv::Vec3i& right) const
	{
		if (left[0] < right[0])
		{
			return true;
		}
		if (left[0] == right[0] && left[1] < right[1])
		{
			return true;
		}
		if (left[0] == right[0] && left[1] == right[1] && left[2] < right[2])
		{
			return true;
		}

		return false;
	}
};

struct MY_COMP_vec4i
{
	bool operator()(const cv::Vec4i& left, const cv::Vec4i& right) const
	{
		if (left[0] < right[0])
		{
			return true;
		}
		if (left[0] == right[0] && left[1] < right[1])
		{
			return true;
		}
		if (left[0] == right[0] && left[1] == right[1] && left[2] < right[2])
		{
			return true;
		}
		if (left[0] == right[0] && left[1] == right[1] && left[2] == right[2] && left[3] < right[3])
		{
			return true;
		}

		return false;
	}
};


//-------------------------------------
//--------disparity computation--------
//-------------------------------------
void stereoMatching(cv::Mat srcLeft, cv::Mat srcRight, cv::Mat& disparityMap,
	StereoMatchingAlgorithms algorithmType, int winSize = 15, int minDisparity = 0, int numDisparity = 64);
void getDisparity_BM(cv::Mat srcLeft, cv::Mat srcRight, cv::Mat& disparityMap, int winSize = 15, int minDisparity = 0, int numDisparity = 64);
void getDisparity_SGBM(cv::Mat srcLeft, cv::Mat srcRight, cv::Mat& disparityMap, int winSize = 15, int minDisparity = 0, int numDisparity = 64);


//-------------------------------------
//--------coat computation-------------
//-------------------------------------
//AD
void computeAD(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_ds,
	DisparityType dispType = DISPARITY_LEFT, int minDisparity = 0, int numDisparity = 30);

//truncated absolute differences TAD C
void computeTAD(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_ds,
	DisparityType dispType = DISPARITY_LEFT, int threshold_T = 30, int minDisparity = 0, int numDisparity = 30);

//cost computation using pixels' color and gradient similarity TAD C+G
void computeSimilarity(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_d_imgs,
	double regularity, double thresC, double thresG, DisparityType dispType,
	int minDisparity, int numDisparity);
void computeSimilarity(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_d_imgs,
	double regularity, double thresC, double thresG, DisparityType dispType,
	int winSize, int minDisparity, int numDisparity);

//squared differences
void computeSD(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_ds, 
	DisparityType dispType = DISPARITY_LEFT, int minDisparity = 0, int numDisparity = 30);

//normalized cross correlation,ncc
void getInputImgNCC(cv::Mat src, std::vector<std::vector<cv::Mat> >& dst, int winSize);
cv::Mat computeNCC(cv::Mat leftImg, cv::Mat rightImg, DisparityType dispType = DISPARITY_LEFT,
	int winSize = 7, int minDisparity = 0, int numDisparity = 30);
void computeNCC(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_ds,
	DisparityType dispType = DISPARITY_LEFT, int winSize = 7,
	int minDisparity = 0, int numDisparity = 30);


//----------------------------------------
//------------cost aggregation------------
//----------------------------------------
//adaptive support weights, asw
cv::Mat computeAdaptiveWeight(cv::Mat leftImg, cv::Mat rightImg, double gamma_c = 30, double gamma_g = 2,
	DisparityType dispType = DISPARITY_LEFT, int winSize = 7, int minDisparity = 186, int numDisparity = 144);
cv::Mat computeAdaptiveWeight_direct8(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType = DISPARITY_LEFT, int winSize = 7, int minDisparity = 186, int numDisparity = 144);

//geodesic distance based asw
float getColorDist(cv::Vec3b pointA, cv::Vec3b pointB);
void getWinGeoDist(cv::Mat originImg, cv::Mat& winDistImg, int winSize = 15, int iterTime = 3);
void getGeodesicDist(cv::Mat originImg, std::map<cv::Point, cv::Mat, MY_COMP_Point2i>& weightGeoDist, int winSize = 15, int iterTime = 3);
cv::Mat computeAdaptiveWeight_geodesic(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType = DISPARITY_LEFT, int winSize = 7, int minDisparity = 186, int numDisparity = 144);

//BLGrid based asw
void createBilGrid(cv::Mat image, std::map<cv::Vec3i, std::pair<double, int>, MY_COMP_vec3i>& bilGrid,
	double sampleRateS = 16, double sampleRateR = 0.07);
void createBilGrid(cv::Mat imageL, cv::Mat imageR, std::map<cv::Vec4i, std::pair<double, int>, MY_COMP_vec4i>& bilGrid,
	int disparity, DisparityType dispType = DISPARITY_LEFT, double sampleRateS = 16, double sampleRateR = 0.07);
void createBilGrid(cv::Mat imageL, cv::Mat imageR, std::map<int, std::map<int, std::map<int, std::map<int, std::pair<double, int> > > > >& bilGrid,
	int disparity, DisparityType dispType = DISPARITY_LEFT, double sampleRateS = 16, double sampleRateR = 0.07);

double trilinear_3d(std::vector<double> axis_diff_xyz, std::vector<double> neighbor_xyz_02);
double quadrlinear_blGrid(std::vector<double> axis_diff_xyzw, std::vector<double> neighbor_xyzw_02);
cv::Mat computeAdaptiveWeight_bilateralGrid(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType = DISPARITY_LEFT, double sampleRateS = 10, double sampleRateR = 10,
	int minDisparity = 186, int numDisparity = 144);

//BLO(1) based asw
cv::Mat getCostSAD_d(cv::Mat leftImg, cv::Mat rightImg, int disparity, DisparityType dispType = DISPARITY_LEFT, int winSize = 35);
cv::Mat computeAdaptiveWeight_BLO1(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType = DISPARITY_LEFT, double sampleRateR = 10, int winSize = 35,
	int minDisparity = 186, int numDisparity = 144);

//guided filter based asw
cv::Mat multiChl_to_oneChl_mul(cv::Mat firstImg, cv::Mat secondImg);
cv::Mat getGuidedFilter(cv::Mat guidedImg, cv::Mat inputP, int r, double eps);
cv::Mat computeAdaptiveWeight_GuidedF(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType = DISPARITY_LEFT, double eps = 0.01, int winSize = 35,
	int minDisparity = 186, int numDisparity = 144);
cv::Mat computeAdaptiveWeight_GuidedF_2(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType = DISPARITY_LEFT, double eps = 0.01, int winSize = 35,
	int minDisparity = 186, int numDisparity = 144);
cv::Mat computeAdaptiveWeight_GuidedF_3(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType = DISPARITY_LEFT, double eps = 1e-6, int winSize = 35,
	int minDisparity = 186, int numDisparity = 144);

//weighted median weight based asw
void computeColorWeightGau(cv::Mat src, std::vector< std::vector<cv::Mat> >& resWins, double rateR, int winSize);
void computeSpaceWeightGau(cv::Mat& dstKernel, int winSize, double rateS);
cv::Mat computeAdaptiveWeight_WeightedMedian(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType = DISPARITY_LEFT, int winSize = 35,
	double sampleRateS = 10, double sampleRateR = 10,
	int minDisparity = 186, int numDisparity = 144);
