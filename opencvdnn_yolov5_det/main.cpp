#include "yolov5-det.h"
int main(int argc, char** argv) {
    std::string onnxpath = "D:/yolo_models/yolov5s.onnx";
    std::string labelpath = "D:/yolo_models/yolo-cls.txt";
    std::shared_ptr<YOLODetector> detector(new YOLODetector());
    detector->initConfig(onnxpath, labelpath);

    cv::VideoCapture capture("D:/yolo_models/play_scoers.mp4");
    cv::Mat frame;
    while (true) {
        capture.read(frame);
        if (frame.empty()) break;
        detector->detect(frame);
        cv::imshow("YOLOv5s + opencvdnn4.11", frame);
        if (cv::waitKey(1) == 27) break; // ESC ÍË³ö
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}