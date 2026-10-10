#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;
using namespace cv;
class YOLO11Detector {
public:
	void initConfig(std::string onnxpath, std::string textpath);
	void detect(cv::Mat& frame);
private:
	float conf_t = 0.25;
	std::vector<std::string> labels;
	int input_w = 640;
	int input_h = 640;
	int output_h;
	int output_w;
	cv::dnn::Net net;
	std::vector<cv::Scalar> color_tables;
};
