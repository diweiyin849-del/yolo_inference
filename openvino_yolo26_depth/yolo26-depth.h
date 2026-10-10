#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;
using namespace cv;
class YOLO11Detector {
public:
	void initConfig(std::string onnxpath);
	void detect(cv::Mat& frame);
private:
	float conf_t = 0.25;
	ov::InferRequest infer_request;
	std::vector<std::string> labels;
	int input_w = 768;
	int input_h = 768;
	int output_h;
	int output_w;
	size_t data_s = input_w * input_h * 3 * sizeof(float);
	std::vector<cv::Scalar> color_tables;
};