#include "rfdetr-det.h"
int main(int argc, char** argv) {
    std::cout << "ORT version: " << Ort::GetVersionString() << std::endl;
    std::string onnxpath = "D:/yolo_models/rfdetr-medium-pill.onnx";
    std::string labelpath = "D:/yolo_models/pill.txt";
    std::shared_ptr<YOLODetector> detector(new YOLODetector());
    detector->initConfig(onnxpath, labelpath);
    cv::VideoCapture capture("C:/Users/24150/Desktop/Frame_404.jpg");
    cv::Mat frame;
    while (true) {
        capture.read(frame);
        if (frame.empty()) break;
        detector->detect(frame);
        cv::imshow("RF-detr + ONNX Runtime1.22", frame);
        if (cv::waitKey(0) == 27) break;
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}