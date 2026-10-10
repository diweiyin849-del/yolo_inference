#include "yolo26-det.h"
int main(int argc, char** argv) {
    std::cout << "ORT version: " << Ort::GetVersionString() << std::endl;
    std::string onnxpath = "D:/yolo_models/yolo26n-100.onnx";
    std::string labelpath = "D:/yolo_models/yolo-cls.txt";
    std::shared_ptr<YOLODetector> detector(new YOLODetector());
    detector->initConfig(onnxpath, labelpath);

    cv::VideoCapture capture("D:/yolo_models/play_scoers.mp4");
    cv::Mat frame;
    while (true) {
        capture.read(frame);
        if (frame.empty()) break;
        detector->detect(frame);
        cv::imshow("YOLO26n + ONNX Runtime1.22", frame);
        if (cv::waitKey(1) == 27) break; 
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}