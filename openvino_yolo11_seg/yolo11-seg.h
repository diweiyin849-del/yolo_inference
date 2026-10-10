#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
using namespace std;
struct DetectResult {
	int classId;
	float conf;
	cv::Rect box;
};

class YOLO11SegDetector {
public:
	void initConfig(std::string onnxpath, std::string textpath);
	void detect(cv::Mat& frame);
private:
	float score_threshold = 0.25;
	std::vector<std::string> labels;
	ov::InferRequest infer_request;
	int input_w = 640;
	int input_h = 640;
	int output_h;
	int output_w;
	int output_mask_weight;
	int output_mask_h;
	int output_mask_w;
	size_t data_s = 640 * 640 * 3 * sizeof(float);
};