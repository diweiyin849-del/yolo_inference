#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudafilters.hpp>
#include<opencv2/cudaarithm.hpp>
#include <fstream>
using namespace std;
using namespace cv;
class YOLO11CallbackAsyncDetector {
public:
	void initConfig(std::string onnxpath, std::string labelpath);
	void detect(std::string source);
private:
	float confidence_t = 0.25;
	size_t data_s = 640 * 640 * 3 * sizeof(float);
	std::vector<ov::InferRequest>request_queue;
	ov::InferRequest request1;
	ov::InferRequest request2;
	ov::InferRequest request3;
	ov::InferRequest request4;
	int numofjobs = 4;
	float xy_factor;
	std::vector<cv::Rect> det_boxes;
	std::vector<int> det_ids;
	std::vector<std::string> classNames;
	void postProcess(ov::Tensor& output);
	void propreoces_frame(ov::Tensor& input_tensor, cv::Mat& img);
};
