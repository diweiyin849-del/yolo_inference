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
	void initConfig(std::string enginefile);
	void detect(cv::Mat& frame, std::vector<DetectResult>& results);
	~YOLOv81112TRTDetector();
private:
	float confidence_threshold = 0.4;
	float score_threshold = 0.25;
	int input_w = 640;
	int input_h = 640;
	int output_h;
	int output_w;
	std::unique_ptr<IRuntime> runtime;
	std::unique_ptr<ICudaEngine> engine;
	std::unique_ptr<IExecutionContext> context;
	void* buffers[2] = { NULL, NULL };
	Mat prob;
	cudaStream_t stream;
};
