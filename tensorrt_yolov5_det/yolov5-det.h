#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudafilters.hpp>
#include<opencv2/cudaarithm.hpp>
#include "NvInfer.h"
using namespace std;
using namespace nvinfer1;
using namespace cv;
// yolov5 yolov6;
struct DetectResult {
	int classId;
	float conf;
	cv::Rect box;
};

class YOLOv5TRTDetector {
public:
	void initConfig(std::string enginefile);
	void detect(cv::Mat& frame, std::vector<DetectResult>& results);
	~YOLOv5TRTDetector();
private:
	float conf_thresholod = 0.25;
	float score_thresholod = 0.25;
	int input_h = 640;
	int input_w = 640;
	int output_h;
	int output_w;
	std::unique_ptr<IRuntime> runtime;
	std::unique_ptr<ICudaEngine> engine;
	std::unique_ptr<IExecutionContext> context;
	void* buffers[2] = { NULL, NULL };
	Mat prob;
	cudaStream_t stream;
};
