#include "yolo11n-pose.h"
int main(int argc, char** argv) {
    std::cout << "ORT version: " << Ort::GetVersionString() << std::endl;
    std::string onnxpath = "D:/yolo_models/yolo11n-pose.onnx";
    std::shared_ptr<YOLODetector> detector(new YOLODetector());
    detector->initConfig(onnxpath);

    cv::VideoCapture capture("D:/yolo_models/QQtest.mp4");
    cv::Mat frame;
    while (true) {
        capture.read(frame);
        if (frame.empty()) break;
        detector->detect(frame);
        cv::imshow("YOLO11-pose + ONNX Runtime1.22", frame);
        if (cv::waitKey(1) == 27) break; 
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}