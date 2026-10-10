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

struct DetectResult {
	int classId;
	float score;
	cv::Rect box;
};

class YOLO11TRTPose {
public:
	void initConfig(std::string enginefile);
	void detect(cv::Mat& frame);
	~YOLO11TRTPose();
private:
	float conf_t = 0.25;
	int input_w;
	int input_h;
	int output_h;
	int output_w;
	std::vector<Scalar> color_tables;
	std::unique_ptr<IRuntime> runtime;
	std::unique_ptr<ICudaEngine> engine;
	std::unique_ptr<IExecutionContext> context;
	void* buffers[2] = { NULL, NULL };
	Mat prob;
	cudaStream_t stream;
};