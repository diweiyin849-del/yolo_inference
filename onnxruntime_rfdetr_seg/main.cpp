#include "rfdetr-seg.h"
int main(int argc, char** argv) {
    std::string onnxpath = "D:/yolo_models/rfdetr-seg-medium.onnx";
    std::string labelpath = "D:/yolo_models/coco.txt";
    std::shared_ptr<YOLODetector> detector(new YOLODetector());
    detector->initConfig(onnxpath, labelpath);

    cv::VideoCapture capture("D:/yolo_models/play_scoers.mp4");
    cv::Mat frame;
    while (true) {
        capture.read(frame);
        if (frame.empty()) break;
        detector->detect(frame);
        cv::imshow("RFDETR-seg + ONNX Runtime1.22 ¼ì²â", frame);
        if (cv::waitKey(1) == 27) break;
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}