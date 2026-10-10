#include "yolo26-obb.h"
int main(int argc, char** argv) {
    std::cout << "ORT version: " << Ort::GetVersionString() << std::endl;
    std::string onnxpath = "D:/yolo_models/yolo26n-obb.onnx";
    std::string labelpath = "D:/yolo_models/dotav1.txt";
    std::unique_ptr< YOLODetector> detector = make_unique< YOLODetector>();
    detector->initConfig(onnxpath, labelpath);

    cv::VideoCapture capture("D:/yolo_models/test_ship.mp4");
    cv::Mat frame;
    while (true) {
        capture.read(frame);
        if (frame.empty()) break;
        detector->detect(frame);
        cv::imshow("YOLO26n-obb + ONNX Runtime1.22", frame);
        if (cv::waitKey(1) == 27) break; 
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}