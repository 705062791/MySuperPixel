#pragma once
#include<opencv2/opencv.hpp>
#include<vector>
#include<limits>
#include<iostream>

using namespace std;
using namespace cv;

//define date structure
typedef struct PixelFeature
{
	int x = -1;
	int y = -1;
	double l = numeric_limits<double>::max();
	double a = numeric_limits<double>::max();
	double b = numeric_limits<double>::max();
};
typedef struct PixelInfo
{
	int label = -1;
	double distance = numeric_limits<double>::max();
};
typedef vector<vector<PixelInfo>> VecPixelInfoMatrix;
typedef vector<vector<PixelFeature>> VecPixelFeaturMatrix;
typedef vector<vector<int>> VecIntMatrix;
typedef vector<vector<double>> VecDoubleMatrix;




class SuperPixel
{
	private:
		VecIntMatrix center_label;
		VecPixelInfoMatrix AllPixelInfo;
		vector<int> center_count;
		vector<PixelFeature> center_info;

		double weight = 0;

		int img_height = -1;
		int img_width = -1;
		int superpixel_num = -1;
		int regression_times = -1;
		int interval = -1;
		int mesh_height = -1;
		int mesh_width = -1;
	public:
		~SuperPixel();
		SuperPixel(int superpixel_num, int img_height, int img_width, double weight, int regression_times);
		double ComputDistance(PixelFeature center_info, PixelFeature Pixel_info);
		template<class T> void RecoverVector(vector<T>& vec);
		void Initialize(cv::Mat image);
		void FindSuperPixcel(cv::Mat image);
		void create_connectivity(cv::Mat image);
		void display(cv::Mat image);
		

};