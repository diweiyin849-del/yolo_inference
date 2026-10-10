#include <fstream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
class YOLO11CallbackAsyncDetector {
public:
	void initConfig(std::string onnxpath, std::string labelpath);
	void detect(std::string source);
private:
	size_t data_s = 3 * 640 * 640 * sizeof(float);
	float confidence_t = 0.25;
	ov::InferRequest curr_request;
	ov::InferRequest next_request;
	float xy_factor;
	std::vector<cv::Rect> det_boxes;
	std::vector<int> det_ids;
	std::vector<std::string> classNames;
	void postProcess(ov::Tensor& output);
	void propreoces_frame(ov::Tensor& input_tensor, cv::Mat& img);
};
