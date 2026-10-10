#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
class YOLO11PoseDetector {
public:
	void initConfig(std::string onnxpath);
	void detect(cv::Mat& frame);
private:
	float score_t = 0.25;
	ov::InferRequest infer_request;
	int input_w = 640;
	int input_h = 640;
	int output_h;
	int output_w;
	size_t data_s = 640 * 640 * 3 * sizeof(float);
	std::vector<cv::Scalar> color_tables;
};
