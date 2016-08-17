#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <iostream>
#include <vector>

int main(void){

	cv::VideoCapture cap(0);
	if (!cap.isOpened()){
		std::cout << "Can't Open camera" << std::endl;
		return -1;
	}
	cv::Mat frame, grayframe;

	int capHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	cv::Mat image = cv::imread("calculator.png"), grayimage;
	cv::namedWindow("Image");

	cv::Size s((int)(image.cols / (float)image.rows * (float)capHeight), capHeight);

	cv::resize(image, image, s);
	cv::cvtColor(image, grayimage, cv::COLOR_BGRA2GRAY);

	std::vector<cv::KeyPoint> imgkeypoint;

	int minHessian = 3000;
	
	cv::SurfFeatureDetector detector(minHessian);
	detector.detect(grayimage, imgkeypoint);

	cv::SurfDescriptorExtractor extractor;
	cv::Mat imagedes;
	extractor.compute(grayimage, imgkeypoint, imagedes);


	

	bool a = true;

	while (a){
		cap >> frame;

		cv::cvtColor(frame, grayframe, cv::COLOR_BGR2GRAY);

		std::vector<cv::KeyPoint> framekeypoint;
		detector.detect(grayframe, framekeypoint);

		cv::Mat framedes;
		extractor.compute(grayframe, framekeypoint, framedes);

		cv::BFMatcher matcher(cv::NORM_L2);
		std::vector<cv::DMatch> matches;

		matcher.match(imagedes, framedes, matches);

		cv::Mat imgmatches;
		cv::drawMatches(image, imgkeypoint, frame, framekeypoint, matches, imgmatches);

		cv::imshow("Image", imgmatches);

		int key;
		switch (key = cv::waitKey(10)){
		case 27: //esc
			a = false;
			break;
		default:
			if (key != -1)std::cout << key << std::endl;
			break;
		}
	}
	return 0;
}