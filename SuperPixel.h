#pragma once
#include<opencv2/opencv.hpp>
#include<vector>
#include<limits>
#include<iostream>

using namespace std;
using namespace cv;

//define date structure
typedef struct PixelInfo
{
	int label = -1;
	double distance = numeric_limits<double>::max();
};
typedef vector<vector<PixelInfo>> VecPixelInfoMatrix;
typedef vector<vector<int>> VecIntMatrix;
typedef vector<vector<double>> VecDoubleMatrix;




class SuperPixel
{
	private:
		SuperPixel(int superpixel_num, int img_height, int img_width, double weight, int regression_times);
		~SuperPixel();

		VecIntMatrix center_label;
		VecPixelInfoMatrix AllPixelInfo;
		vector<int> center_x;
		vector<int> center_y;

		double weight = 0;

		int img_height = -1;
		int img_width = -1;
		int superpixel_num = -1;
		int regression_times = -1;
		int interval = -1;
		int mesh_height = -1;
		int mesh_width = -1;
	public:
		void Initialize();
		void FindSuperPixcel(cv::Mat image);
		void ComputDistance();
};