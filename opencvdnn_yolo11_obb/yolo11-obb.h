#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
class YOLO11ObbDetector {
public:
	void initConfig(std::string onnxpath);
	void detect(cv::Mat& frame, std::vector<std::string>& classNames);
	void getRotatedPts(std::vector<cv::Point>& rrts, float angle, std::vector<cv::Point>& box_pts, cv::Rect& box);
private:
	float score_threshold = 0.25;
	int input_w = 1024;
	int input_h = 1024;
	int output_h;
	int output_w;
	cv::dnn::Net net;
};
