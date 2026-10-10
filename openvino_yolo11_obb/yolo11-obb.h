#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudafilters.hpp>
#include<opencv2/cudaarithm.hpp>
#include <fstream>
using namespace cv;
class YOLO11OBBDetector {
public:
	void initConfig(std::string onnxpath, std::string textpath);
	void detect(cv::Mat& frame);
	void getRotatedPts(std::vector<cv::Point>& rrts, float angle, std::vector<cv::Point>& box_pts, cv::Rect& box);
private:
	double t_score = 0.25;
	ov::InferRequest infer_request;
	std::vector<std::string> labels;
	int input_w = 1024;
	int input_h = 1024;
	int output_h;
	int output_w;
	size_t data_s = 1024 * 1024 * 3 * sizeof(float);
	std::vector<cv::Scalar> color_tables;
};
