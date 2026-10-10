#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudafilters.hpp>
#include<opencv2/cudaarithm.hpp>
#include "NvInfer.h"

using namespace nvinfer1;
using namespace cv;
using namespace std;
struct DetectResult {
	int classId;
	float conf;
	cv::Rect box;
};

class YOLOv8TRTSegment {
public:
	void initConfig(std::string enginefile, float conf_thresholod, float score_thresholod);
	void detect(cv::Mat& frame, std::vector<DetectResult>& results);
	~YOLOv8TRTSegment();
private:
	float conf_thresholod = 0.25;
	float score_thresholod = 0.25;
	int input_h = 640;
	int input_w = 640;
	int output_h;
	int output_w;
	int output_mask_weight;
	int output_mask_h;
	int output_mask_w;
	std::unique_ptr<IRuntime> runtime;
	std::unique_ptr<ICudaEngine> engine;
	std::unique_ptr<IExecutionContext> context;
	void* buffers[3] = { NULL, NULL , NULL };
	Mat prob; //detect
	Mat mprob; // mask
	cudaStream_t stream;
};
