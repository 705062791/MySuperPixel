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
	RecoverVector(center_label);
	RecoverVector(AllPixelInfo);
	RecoverVector(center_count);
	RecoverVector(center_info);

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

			single_pixel.l = Lab_image.at<Vec3b>(i, j)[0]*100/255;
			single_pixel.a = Lab_image.at<Vec3b>(i, j)[1]-128;
			single_pixel.b = Lab_image.at<Vec3b>(i, j)[2]-128;

			center_info.push_back(single_pixel);

		}
	}


	//initialize the pixelinfo
	PixelInfo single_pixel_info;
	AllPixelInfo.resize(img_height);
	
	for (int i = 0; i < img_height; i++)
	{
		AllPixelInfo[i].resize(img_width);
	}

	cout << "Initialize over" << endl;
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
	clock_t part_1_start;
	clock_t part_1_end;

	clock_t part_2_start;
	clock_t part_2_end;

	Mat Lab_image;

	cv::cvtColor(image, Lab_image, CV_BGR2Lab);

	for (int i = 0; i < this->regression_times; i++)
	{
		part_1_start = clock();
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
							Lab_image.at<cv::Vec3b>(m,n)[0] * 100 / 255,
							Lab_image.at<cv::Vec3b>(m,n)[1] - 128,
							Lab_image.at<cv::Vec3b>(m,n)[2] - 128
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
		part_1_end = clock();


		part_2_start = clock();
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
			center_info[j].l = Lab_image.at<cv::Vec3b>(new_y, new_x)[0] * 100 / 255;
			center_info[j].a = Lab_image.at<cv::Vec3b>(new_y, new_x)[1] - 128;
			center_info[j].b = Lab_image.at<cv::Vec3b>(new_y, new_x)[2] - 128;

		}

		part_2_end = clock();

	}
	printf("part 1 cost %f ms\n", double(part_1_end) - part_1_start);
	printf("part 2 cost %f ms\n", double(part_2_end) - part_2_start);

}

void SuperPixel::create_connectivity(cv::Mat image)
{
	int neighbor_label = 0;
	cv::Mat Lab_image;
	cv::cvtColor(image, Lab_image, CV_BGR2Lab);
	
	//建立一个掩码的矩阵，标记是否重新分类
	cv::Mat AllPixelInfoMask = cv::Mat::zeros(cv::Size(img_width, img_height), CV_8U);

	//建立一个4通量的矩阵
	const int dx4[4] = {-1 ,  0 , 1 , 0 };
	const int dy4[4] = { 0 , -1 , 0 , 1 };

	//设置阈值
	const int lims = (img_height * img_width) / (int)(center_info.size());

	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{
			//如果没有被分类
			if (AllPixelInfoMask.at<uchar>(i, j) == 0)
			{
				
				vector<cv::Point2i> elements;
				elements.push_back(cv::Point2i(j, i));
				AllPixelInfoMask.at<uchar>(i, j) = 1;

				//查看上下左右4个分量
				for (int m = 0; m < 4; m++)
				{
					int x = elements[0].x + dx4[m];
					int y = elements[0].y + dy4[m];

					//判断是否越过了边界
					if (x < img_width && x >= 0 && y < img_height && y >= 0)
					{
						//如果该联通分量已经有了类别
						if (AllPixelInfoMask.at<uchar>(y, x) != 0)
						{
							neighbor_label = AllPixelInfo[y][x].label;
						}
					}
				}


				//开始查找周围相同标签的像素
				int count = 1;
				for (int m = 0; m < count; m++)
				{
					for (int n = 0; n < 4; n++)
					{
						int x = elements[m].x + dx4[n];
						int y = elements[m].y + dy4[n];

						if (x < img_width && x >= 0 && y < img_height && y >= 0)
						{
							if (AllPixelInfo[i][j].label == AllPixelInfo[y][x].label&& AllPixelInfoMask.at<uchar>(y, x)==0)
							{
								elements.push_back(cv::Point2i(x, y));
								AllPixelInfoMask.at<uchar>(y, x) = 1;
								count += 1;
								
							}
						}

					}
				}

				if (count <= lims >> 2)
				{
					for (int m = 0; m < count; m++)
					{
						AllPixelInfo[elements[m].y][elements[m].x].label = neighbor_label;
					}
				}
			    
			}
		}
	}	


}

void SuperPixel::display(cv::Mat image)
{
	int dx8[8] = { -1 , 0 , 1 , 1 , 1 ,  0 , -1 , -1 };
	int dy8[8] = {  1 , 1 , 1 , 0 ,-1 , -1 , -1 ,  0 };

	vector<cv::Point2i> coutour_pixel;
	Mat color_mask = cv::Mat::zeros(cv::Size(img_width, img_height), CV_8U);

	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{
			int count = 0;

			for (int m = 0; m < 8; m++)
			{
				int x = j + dx8[m];
				int y = i + dy8[m];

				if (x >= 0 && x < img_width && y >= 0 && y < img_height)
				{
					if (color_mask.at<uchar>(y, x) == 0 && AllPixelInfo[i][j].label != AllPixelInfo[y][x].label)
					{
						count++;
					}
				}
			}

			if (count >= 2)
			{
				color_mask.at<uchar>(i, j) = 1;
				coutour_pixel.push_back(cv::Point2i(j,i));
			}

		}
	}


	for (int i = 0; i < coutour_pixel.size(); i++)
	{
		image.at<Vec3b>(coutour_pixel[i].y, coutour_pixel[i].x) = Vec3b(255, 255, 255);
	}

	cv::namedWindow("SuperPixel");
	cv::imshow("SuperPixel", image);
	cv::waitKey(0);

}