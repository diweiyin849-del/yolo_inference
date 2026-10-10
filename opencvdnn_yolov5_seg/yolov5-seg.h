#include <iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudafilters.hpp>
#include<opencv2/cudaarithm.hpp>
#include<fstream>
using namespace std;
using namespace cv;
using namespace std;
using namespace cv;
class YOLODetector {
public:
    void initConfig(std::string onnxpath, std::string textpath);
    void detect(cv::Mat& frame);
private:
    float conf_t = 0.25;          // 置信度阈值
    int input_w = 640;            // 模型输入宽度
    int input_h = 640;            // 模型输入高度
    int output_h;
    int output_w;
    int output_mask_weight;
    int output_mask_h;
    int output_mask_w;
    std::vector<std::string> labels;          // 类别名称
    std::vector<cv::Scalar> color_tables;     // 预留（可不用）
    cv::dnn::Net net;
};

