#include"SuperPixel.h"
using namespace std;
using namespace cv;

SuperPixel::SuperPixel(int superpixel_num, int img_height, int img_width, double weight, int regression_times)
{
	this->superpixel_num = superpixel_num;
	this->img_height = img_height;
	this->img_width = img_width;
	this->weight = weight;
	this->regression_times = regression_times;

	//comput the interval
	this->interval = int(sqrt(img_height * img_width / superpixel_num));
	this->mesh_height = int(img_height / interval);
	this->mesh_width = int(img_width / interval);


}

SuperPixel::~SuperPixel()
{

}

void SuperPixel::Initialize()
{
	this->center_x.clear();
	this->center_y.clear();
	//initialize the center
	for (int i = int(this->interval / 2-1); i < img_height; i += this->interval)
	{
		for (int j = int(this->interval / 2-1); j < img_width; j += this->interval)
		{
			this->center_x.push_back(j);
			this->center_y.push_back(i);
		}
	}

	//initialize the pixelinfo
	PixelInfo single_pixel_info;

	for (int i = 0; i < this->img_height; i++)
	{
		vector<PixelInfo> row;
		for (int j = 0; j < this->img_width; j++)
		{
			row.push_back(single_pixel_info);
		}
		this->AllPixelInfo.push_back(row);
		row.clear();
	}

}

void SuperPixel::FindSuperPixcel(cv::Mat image)
{


	for (int i = 0; i < this->regression_times; i++)
	{
		//comput the d in (2*interval)*(2*interval) range
		for (int j = 0; j < center_x.size(); j++)
		{

		}
	}
}