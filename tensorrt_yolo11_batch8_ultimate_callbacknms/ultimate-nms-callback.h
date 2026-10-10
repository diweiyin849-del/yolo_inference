#ifndef ULTIMATE_NMS_CALLBACK_H
#define ULTIMATE_NMS_CALLBACK_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "NvInfer.h"

using namespace nvinfer1;
using namespace cv;
using namespace std;

struct DetectResult {
    int classId;
    float score;
    cv::Rect box;
};

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

class YOLOv81112TRTDetector {
public:
    YOLOv81112TRTDetector();
    ~YOLOv81112TRTDetector();

    void initConfig(std::string enginefile);

    void detect(cv::Mat& frame, std::vector<DetectResult>& results);

    using PostProcessCallback = std::function<void(const std::vector<std::vector<DetectResult>>&)>;
    void detectBatch(const std::vector<cv::Mat>& frames,
        std::vector<std::vector<DetectResult>>& batch_results,
        PostProcessCallback callback = nullptr);

private:
    float confidence_threshold = 0.4;
    float score_threshold = 0.25;
    int input_w = 640;
    int input_h = 640;
    int output_h;
    int output_w;
    int m_batchSize = 16;

    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    void* buffers[2] = { nullptr, nullptr };
    cv::Mat prob;          // 同步模式用
    cudaStream_t stream;

    struct PostTask {
        cv::Mat prob_data;          // 异步模式专用，独立的内存块
        std::vector<float> x_factors, y_factors;
        PostProcessCallback callback;
        int output_h;
        int output_w;
        int batch_size;
        cudaStream_t stream;
    };

    std::queue<PostTask> post_queue_;
    std::mutex post_mutex_;
    std::condition_variable post_cv_;
    std::thread post_thread_;
    bool stop_post_ = false;

    void postProcessingWorker();
    void runPostProcessing(const PostTask& task);
    void syncPostProcessing(const cv::Mat& prob_data,
        const std::vector<float>& x_factors,
        const std::vector<float>& y_factors,
        std::vector<std::vector<DetectResult>>& results);
};

#endif