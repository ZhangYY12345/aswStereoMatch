// aswStereoMatch.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "methods/aswMethods.h"

using namespace cv;

extern std::map<StereoMatchingAlgorithms, std::string> algorithmFlags;

int main()
{
	//Mat stereoPair_rectified_left = imread("D:/studying/stereo vision/research code/data/cones-png-2/cones/im2.png");
	//Mat stereoPair_rectified_right = imread("D:/studying/stereo vision/research code/data/cones-png-2/cones/im6.png");

	//Mat stereoPair_rectified_left = imread("D:\\studying\\stereo vision\\research code\\data\\CALIB20190702\\rectifiedLeft.jpg");
	//Mat stereoPair_rectified_right = imread("D:\\studying\\stereo vision\\research code\\data\\CALIB20190702\\rectifiedRight.jpg");

	std::string imgPath = "D:\\studying\\stereo vision\\research code\\data\\2019-07-23\\rectify";
	//load all the images in the folder
	cv::String filePath = imgPath + "\\*L_rectify.jpg";
	std::vector<cv::String> fileNames;
	cv::glob(filePath, fileNames, false);

	for (int i = 0; i < fileNames.size(); i++)
	{
		Mat stereoPair_rectified_left = imread(fileNames[i]);
		Mat stereoPair_rectified_right = imread(fileNames[i].substr(0, fileNames[i].length()-13) + "R_rectify.jpg");

		resize(stereoPair_rectified_left, stereoPair_rectified_left, Size(640, 360));
		resize(stereoPair_rectified_right, stereoPair_rectified_right, Size(640, 360));

		Mat sobelL_x, sobelL_y, sobelR_x, sobelR_y;
		Sobel(stereoPair_rectified_left, sobelL_x, CV_8U, 1, 0, 3, 1, 0, BORDER_REFLECT);
		Sobel(stereoPair_rectified_left, sobelL_y, CV_8U, 0, 1, 3, 1, 0, BORDER_REFLECT);

		Sobel(stereoPair_rectified_right, sobelR_x, CV_8U, 1, 0, 3, 1, 0, BORDER_REFLECT);
		Sobel(stereoPair_rectified_right, sobelR_y, CV_8U, 0, 1, 3, 1, 0, BORDER_REFLECT);

		stereoPair_rectified_left = stereoPair_rectified_left + sobelL_x + sobelL_y;
		stereoPair_rectified_right = stereoPair_rectified_right + sobelR_x + sobelR_y;

		for (auto itor = algorithmFlags.begin(); itor != algorithmFlags.end(); itor++)
		{
			Mat disparityMap;
			stereoMatching(stereoPair_rectified_left, stereoPair_rectified_right, disparityMap, DISPARITY_LEFT, itor->first, 15, 0, 64);
			imwrite(fileNames[i].substr(0, fileNames[i].length() - 13) + "_" + itor->second + "disparity.jpg", disparityMap);

			disparityMap.convertTo(disparityMap, CV_8UC1);
			normalize(disparityMap, disparityMap, 0, 255, NORM_MINMAX);

			imwrite(fileNames[i].substr(0, fileNames[i].length() - 13) + "_" + itor->second + "disparity_0255.jpg", disparityMap);
		}
	}
	waitKey(0);
	return 0;
}
