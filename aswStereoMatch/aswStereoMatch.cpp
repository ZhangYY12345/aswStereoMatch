// aswStereoMatch.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "methods/aswMethods.h"

using namespace cv;

int main()
{
	Mat stereoPair_rectified_left = imread("D:/studying/stereo vision/research code/data/cones-png-2/cones/im2.png");
	Mat stereoPair_rectified_right = imread("D:/studying/stereo vision/research code/data/cones-png-2/cones/im6.png");

	Mat disparityMap;
	stereoMatching(stereoPair_rectified_left, stereoPair_rectified_right, disparityMap, ADAPTIVE_WEIGHT_GUIDED_FILTER);
	imwrite("disparity.jpg", disparityMap);

	normalize(disparityMap, disparityMap, 0, 255, NORM_MINMAX);
	imwrite("disparity_0255.jpg", disparityMap);

	waitKey(0);
	return 0;
}
