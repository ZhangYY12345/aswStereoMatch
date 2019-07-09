// aswStereoMatch.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "methods/aswMethods.h"

using namespace cv;

int main()
{
	//Mat stereoPair_rectified_left = imread("D:/studying/stereo vision/research code/data/cones-png-2/cones/im2.png");
	//Mat stereoPair_rectified_right = imread("D:/studying/stereo vision/research code/data/cones-png-2/cones/im6.png");

	Mat stereoPair_rectified_left = imread("D:\\studying\\stereo vision\\research code\\data\\CALIB20190702\\rectifiedLeft.jpg");
	Mat stereoPair_rectified_right = imread("D:\\studying\\stereo vision\\research code\\data\\CALIB20190702\\rectifiedRight.jpg");

	Mat sobelL_x, sobelL_y, sobelR_x, sobelR_y;
	Sobel(stereoPair_rectified_left, sobelL_x, CV_8U, 1, 0,3,1, 0, BORDER_REFLECT);
	Sobel(stereoPair_rectified_left, sobelL_y, CV_8U, 0, 1, 3, 1, 0, BORDER_REFLECT);

	Sobel(stereoPair_rectified_right, sobelR_x, CV_8U, 1, 0, 3, 1, 0, BORDER_REFLECT);
	Sobel(stereoPair_rectified_right, sobelR_y, CV_8U, 0, 1, 3, 1, 0, BORDER_REFLECT);

	stereoPair_rectified_left = stereoPair_rectified_left + sobelL_x + sobelL_y;
	stereoPair_rectified_right = stereoPair_rectified_right + sobelR_x + sobelR_y;

	Mat disparityMap;
	stereoMatching(stereoPair_rectified_left, stereoPair_rectified_right, disparityMap, SGBM,7, 20, 64);
	imwrite("disparity.jpg", disparityMap);

	normalize(disparityMap, disparityMap, 0, 255, NORM_MINMAX);
	imwrite("disparity_0255.jpg", disparityMap);

	waitKey(0);
	return 0;
}
