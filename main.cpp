#include"SuperPixel.h"
#include<opencv2/core.hpp>
#include<opencv2/ximgproc.hpp>
#include<ctime>
using namespace std;
using namespace cv;

int main()
{
	//time count
	clock_t start;
	clock_t end;
	double totall;

	cv::Mat image = cv::imread("test.jpg");
	cv::Mat bilateral_image;
	cv::bilateralFilter(image, bilateral_image, 25, 50, 25/2);


	SuperPixel super_pixel(216,720,1080,32,4);

	start = clock();
	super_pixel.Initialize(bilateral_image);
	super_pixel.FindSuperPixcel(bilateral_image);
	super_pixel.create_connectivity(bilateral_image);
	end = clock();

	totall = double(end) - start;

	printf("program totall cost %f", totall);

	super_pixel.display(bilateral_image);

}

//int main()
//{
//	cv::Mat image = cv::imread("test.jpg");
//	cv::Mat labels, mask;
//
//	Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(image,101,20);
//	slic->iterate(10);
//	slic->enforceLabelConnectivity();
//	slic->getLabelContourMask(mask);
//	slic->getLabels(labels);
//	
//	int number = slic->getNumberOfSuperpixels();
//
//	cout << number << endl;
//
//	image.setTo(Scalar(255, 255, 255), mask);
//
//	cv::namedWindow("test");
//	cv::imshow("test", image);
//	cv::waitKey(0);
//
//}