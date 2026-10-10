#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
using namespace std;
using namespace cv;
class YOLO11Detector {
public:
	void initConfig(std::string onnxpath, std::string textpath);
	void detect(cv::Mat& frame);
	double sigmoid_stable(double x);
private:
	float conf_t = 0.25;
	ov::InferRequest infer_request;
	std::vector<std::string> labels;
	int input_w = 576;
	int input_h = 576;
	int output_det_h;
	int output_det_w;
	int output_label_h;
	int output_label_w;
	size_t data_s = input_w * input_h * 3 * sizeof(float);
	std::vector<cv::Scalar> color_tables;
};

