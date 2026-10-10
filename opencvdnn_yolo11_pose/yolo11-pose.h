#include <opencv2/opencv.hpp>
class YOLO11PoseDetector {
public:
	void initConfig(std::string onnxpath);
	void detect(cv::Mat& frame);
private:
	float score_t = 0.25;
	int input_w = 640;
	int input_h = 640;
	int output_h;
	int output_w;
	cv::dnn::Net net;
	std::vector<cv::Scalar> color_tables;
};

