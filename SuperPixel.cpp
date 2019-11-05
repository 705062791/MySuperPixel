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

template<class T> void SuperPixel::RecoverVector(vector<T>& vec)
{
	vector<T> tempVec;
	tempVec.swap(vec);
}

void SuperPixel::Initialize(cv::Mat image)
{
	Mat Lab_image;

	cv::cvtColor(image, Lab_image, CV_BGR2Lab);

	//initialize the center
	for (int i = int(this->interval / 2-1); i < img_height; i += this->interval)
	{
		for (int j = int(this->interval / 2-1); j < img_width; j += this->interval)
		{
			PixelFeature single_pixel;
			single_pixel.x = j;
			single_pixel.y = i;

			single_pixel.l = Lab_image.at<Vec3i>(i, j)[0];
			single_pixel.a = Lab_image.at<Vec3i>(i, j)[1];
			single_pixel.b = Lab_image.at<Vec3i>(i, j)[2];

			center_info.push_back(single_pixel);

		}
	}


	//initialize the pixelinfo
	PixelInfo single_pixel_info;
	AllPixelInfo.resize(img_height);
	
	for (int i = 0; i < img_height; i++)
	{
		AllPixelInfo.resize(img_width);
	}

	//for (int i = 0; i < this->img_height; i++)
	//{
	//	for (int j = 0; j < this->img_width; j++)
	//	{
	//		AllPixelInfo[i][j]=single_pixel_info;
	//	}

	//}

}

double SuperPixel::ComputDistance(PixelFeature center_info, PixelFeature Pixel_info)
{
	double residual_lab = pow(center_info.l - Pixel_info.l, 2) + pow(center_info.a - Pixel_info.a, 2) + pow(center_info.b - Pixel_info.b, 2);
	double residual_x_y = pow(center_info.x - Pixel_info.x, 2) + pow(center_info.y - Pixel_info.y, 2);

	double D = residual_lab + residual_x_y*(weight/interval);

	return D;
}

void SuperPixel::FindSuperPixcel(cv::Mat image)
{
	Mat Lab_image;

	cv::cvtColor(image, Lab_image, CV_BGR2Lab);

	for (int i = 0; i < this->regression_times; i++)
	{
		//process the neighbor pixel information double interval*double interval
		for (int j = 0; j < this->mesh_height; j++)
		{
			for (int k = 0; k < this->mesh_width; k++)
			{
				int x = center_info[double(j)*this->mesh_width + k].x;
				int y = center_info[double(j)*this->mesh_width + k].y;
				
				int min_x = x - 3 / 2 * this->interval >= 0 ? int(x - 3 / 2 * this->interval) : 0;
				int max_x = x + 3 / 2 * this->interval < this->img_width ? int(x + 3 / 2 * this->interval) : this->img_width;

				int min_y = y - 3 / 2 * this->interval >= 0 ? int(y - 3 / 2 * this->interval) : 0;
				int max_y = y + 3 / 2 * this->interval < this->img_height ? int(y + 3 / 2 * this->interval) : this->img_height;

				for (int m = min_y; m < max_y; m++)
				{
					for (int n = min_x; n < max_x; n++)
					{
						//comput distance
						PixelFeature single_pixel =
						{
							n,m,
							Lab_image.at<cv::Vec3i>(m,n)[0],
							Lab_image.at<cv::Vec3i>(m,n)[1],
							Lab_image.at<cv::Vec3i>(m,n)[2]
						};

						double D = ComputDistance(center_info[double(j) * mesh_width + k],single_pixel);

						if(D<= AllPixelInfo[m][n].distance)
						{
							AllPixelInfo[m][n].distance = D;
							AllPixelInfo[m][n].label = double(j) * mesh_width + k;
						}
					}
				}

			}
		}

		//initial center count
		center_count.clear();
		center_count.resize(int(double(mesh_height) * mesh_width));
		for (int j = 0; j < center_info.size(); j++)
		{
			center_count[j] = 0;
		}

		//update center
		for (int j = 0 ; j < img_height; j++)
		{
			for (int k = 0 ; k < img_width; k++)
			{
				int center_label = AllPixelInfo[j][k].label;
				center_count[center_label]++;

				center_info[center_label].x += k;
				center_info[center_label].y += j;
			}
		}

		for (int j = 0; j < center_info.size(); j++)
		{
			int new_x = int(center_info[j].x / center_count[j]);
			int new_y = int(center_info[j].y / center_count[j]);

			center_info[j].x = new_x;
			center_info[j].y = new_y;
			center_info[j].l = Lab_image.at<cv::Vec3i>(new_y, new_x)[0];
			center_info[j].a = Lab_image.at<cv::Vec3i>(new_y, new_x)[1];
			center_info[j].b = Lab_image.at<cv::Vec3i>(new_y, new_x)[2];

		}

	}
}