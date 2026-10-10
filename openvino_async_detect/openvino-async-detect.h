#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
using namespace std;
using namespace cv;
class YOLO11AsyncDetector {
public:
	void initConfig(std::string onnxpath, std::string textpath);
	void detect(std::string videofile);
private:
	float score_t = 0.25;
	std::vector<ov::InferRequest> inferreqs;
	std::vector<std::string> labels;
	int input_w = 640;
	int input_h = 640;
	int output_h;
	int output_w;
	size_t data_s = 640 * 640 * 3 * sizeof(float);
	std::vector<cv::Scalar> color_tables;
	void preprocess(cv::Mat& frame, ov::Tensor& input_tensor, float& xy_factor);
	void postProcess(cv::Mat& detout, cv::Mat& frame, float xy_factor);
};
