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
		//process the neighbor pixel information double interval*double interval
		for (int j = 0; j < this->mesh_height; j++)
		{
			for (int k = 0; k < this->mesh_width; k++)
			{
				int x = this->center_x[j*this->mesh_width + k];
				int y = this->center_y[j*this->mesh_width + k];
				
				int min_x = x - 3 / 2 * this->interval >= 0 ? int(x - 3 / 2 * this->interval) : 0;
				int max_x = x + 3 / 2 * this->interval < this->img_width ? int(x + 3 / 2 * this->interval) : this->img_width;

				int min_y = y - 3 / 2 * this->interval >= 0 ? int(y - 3 / 2 * this->interval) : 0;
				int max_y = y + 3 / 2 * this->interval < this->img_height ? int(y + 3 / 2 * this->interval) : this->img_height;

				for (int m = min_y; m < max_y; m++)
				{
					for (int n = min_x; n < max_x; n++)
					{
						//comput distance

					}
				}

			}
		}
	}
}