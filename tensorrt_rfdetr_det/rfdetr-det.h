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
class YOLOv81112TRTDetector {
public:
	void initConfig(std::string enginefile, std::string label_map);
	void detect(cv::Mat& frame);
	double sigmoid_stable(double x);
	~YOLOv81112TRTDetector();
private:
	float score_threshold = 0.25;
	int input_w = 576;
	int input_h = 576;
	int output_det_h;
	int output_det_w;
	int output_label_h;
	int output_label_w;
	std::vector<string> classNames;
	std::unique_ptr<IRuntime> runtime;
	std::unique_ptr<ICudaEngine> engine;
	std::unique_ptr<IExecutionContext> context;
	void* buffers[3] = { NULL, NULL };
	Mat det_prob;
	Mat label_prob;
	cudaStream_t stream;
};

