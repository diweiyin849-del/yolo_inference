#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
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
    std::vector<std::string> labels;
    std::vector<cv::Scalar> color_tables;
    ov::InferRequest infer_request;
    size_t data_s = 640 * 640 * 3 * sizeof(float);
};
