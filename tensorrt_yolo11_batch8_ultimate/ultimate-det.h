
#ifndef YOLO11_BATCH8_DET_H
#define YOLO11_BATCH8_DET_H
#include <opencv2/core/cuda_stream_accessor.hpp>
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
    float score;
    cv::Rect box;
};

class YOLOv81112TRTDetector {
public:

    ~YOLOv81112TRTDetector();
    void initConfig(std::string enginefile);
    // 单帧推理(不够batchsize时)
    void detect(cv::Mat& frame, std::vector<DetectResult>& results);
    // 批量推理：一次处理 batchSize 帧
    void detectBatch(const std::vector<cv::Mat>& frames, std::vector<std::vector<DetectResult>>& batch_results);

private:
    float confidence_threshold = 0.4;
    float score_threshold = 0.25;
    int input_w = 640;
    int input_h = 640;
    int output_h;          // 每帧输出的行数（例如 84）
    int output_w;          // 每帧输出的列数（例如 8400）
    int m_batchSize = 16;   // 从 engine 读取的实际 batch 大小

    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    void* buffers[2] = { nullptr, nullptr };
    Mat prob;              // CPU 端输出缓存（大小 = batch * output_h * output_w）
    cudaStream_t stream;
};

#endif