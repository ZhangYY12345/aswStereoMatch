#pragma once
#include <opencv2/opencv.hpp>

enum DisparityType
{
	DISPARITY_LEFT = 0,
	DISPARITY_RIGHT = 1,
};

enum StereoMatchingAlgorithms
{
	BM = 0,
	SGBM = 1,
	ADAPTIVE_WEIGHT = 2,
	ADAPTIVE_WEIGHT_8DIRECT = 3,
	ADAPTIVE_WEIGHT_GEODESIC = 4,
	ADAPTIVE_WEIGHT_BILATERAL_GRID = 5,
	ADAPTIVE_WEIGHT_BLO1 = 6,
	ADAPTIVE_WEIGHT_GUIDED_FILTER = 7,
	ADAPTIVE_WEIGHT_GUIDED_FILTER_2 = 8,
	ADAPTIVE_WEIGHT_GUIDED_FILTER_3 = 9,
	ADAPTIVE_WEIGHT_MEDIAN = 10,
};

enum PCLFILTERS_
{
	PASS_THROUGH = 0,					//ʹ��ֱͨ�˲����Ե��ƽ����˲�����
	VOXEL_GRID = 1,						//ʹ��VoxelGrid�˲��������²����������������е��������������ʾ������������
	STATISTIC_OUTLIERS_REMOVE = 2,		//ʹ��StatisticalOutlierRemoval�˲����Ƴ���Ⱥ��
	MODEL_COEFFICIENTS = 3,				//����ͶӰ��һ��������ģ���ϣ�����ƽ�����ȣ�
	EXTRACT_INDICES = 4,				//ʹ��ExtractIndices�˲���������ĳһ�ָ��㷨��ȡ�����е�һ���Ӽ�
	CONDITIONAL_REMOVAL = 5,			//ʹ��CondidtionalRemoval�˲�����һ��ɾ�����������ĵ����趨��һ����������ָ����������ݵ�
	RADIUS_OUTLIER_REMOVAL = 6,			//ʹ��RadiusOutlierRemoval�˲�����ɾ��������ĵ���һ����Χ��û�дﵽ�㹻����������������ݵ�
	CROP_HULL= 7,						//ʹ��CropHull�˲����õ�2D��ն�����ڲ������ⲿ�ĵ���
};

enum CONSENSUS_MODEL_TYPE_
{
	CONSENSUS_MODEL_SPHERE_ = 0,
	CONSENSUS_MODEL_PLANE_ = 1,
};

bool check_image(const cv::Mat &image, std::string name = "Image");
bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2);

cv::Point2d PointF2D(cv::Point2f point);
std::vector<cv::Point2d> VecPointF2D(std::vector<cv::Point2f> pts);


template<typename _Tp>
std::vector<_Tp> convertMat2Vector(const cv::Mat mat)
{
	if (mat.isContinuous())
	{
		return (std::vector<_Tp>)(mat.reshape(0, 1));
	}

	cv::Mat mat_ = mat.clone();
	std::vector<_Tp> vecMat = mat_.reshape(0, 1);
	return (std::vector<_Tp>)(mat_.reshape(0, 1));
}