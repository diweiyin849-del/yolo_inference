#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;
using namespace std;
struct DetectResult {
	int classId;
	float score;
	cv::Rect box;
};

class YOLOv81112TRTDetector {
public:
	void initConfig(std::string enginefile, std::string label_map);
	void detect(cv::Mat& frame);
	void getRotatedPts(std::vector<cv::Point>& rrts, float angle, std::vector<cv::Point>& box_pts, cv::Rect& box);
	~YOLOv81112TRTDetector();
private:
	float score_threshold = 0.25;
	int input_w = 1024;
	int input_h = 1024;
	int output_h;
	int output_w;
	std::vector<string> classNames;
	std::unique_ptr<IRuntime> runtime;
	std::unique_ptr<ICudaEngine> engine;
	std::unique_ptr<IExecutionContext> context;
	void* buffers[2] = { NULL, NULL };
	Mat prob;
	cudaStream_t stream;
};
