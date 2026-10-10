#include "onnxruntime_cxx_api.h"
#include"cpu_provider_factory.h"
#include <iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudafilters.hpp>
#include<opencv2/cudaarithm.hpp>
#include<fstream>
using namespace std;
using namespace cv;
class YOLODetector {
public:
    void initConfig(std::string onnxpath, std::string textpath);
    void detect(cv::Mat& frame);
    double sigmoid_stable(double x);
    ~YOLODetector();

private:
    float conf_t = 0.25;          // 置信度阈值
    int input_w = 432;            // 模型输入宽度
    int input_h = 432;            // 模型输入高度
    int output_det_h;
    int output_det_w;
    int output_label_h;
    int output_label_w;
    int output_mask_weight;
    int output_mask_h;
    int output_mask_w;
    std::vector<std::string> labels;          // 类别名称
    std::vector<cv::Scalar> color_tables;     // 预留
    // ONNX Runtime 相关成员
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "RFDETR-seg-Onnx");
    Ort::SessionOptions session_options;
    Ort::Session session{ nullptr };
    std::vector<std::string>input_node_names; //用于存储输入的名称
    std::vector<std::string>output_node_names;//用于存储输出的名称
    std::array<const char*, 1>inputNames; //准备输入名称数组
    std::array<const char*, 3>outputNames;//准备输名称数组
};