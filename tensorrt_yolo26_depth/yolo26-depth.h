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
	void initConfig(std::string enginefile);
	void detect(cv::Mat& frame);
	~YOLOv81112TRTDetector();
private:
	float score_threshold = 0.25;
	int input_w = 768;
	int input_h = 768;
	int output_h;
	int output_w;
	size_t data_s = input_w * input_h * 3 * sizeof(float);
	std::vector<string> classNames;
	std::unique_ptr<IRuntime> runtime;
	std::unique_ptr<ICudaEngine> engine;
	std::unique_ptr<IExecutionContext> context;
	void* buffers[2] = { NULL, NULL };
	Mat prob;
	cudaStream_t stream;
};

