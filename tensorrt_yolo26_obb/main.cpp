#include"yolo26-obb.h"
using namespace std;
std::string label_map = "D:/yolo_models/dotav1.txt";
int main(int argc, char** argv) {
	std::unique_ptr<YOLOv26TRTDetector> detector = std::make_unique<YOLOv26TRTDetector>();
	detector->initConfig("D:/TensorRT-10.8.0.43/bin/yolo26n-obb-fp16.engine", label_map);
	cv::VideoCapture capture("D:/yolo_models/test_ship.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("YOLO26n-obb + TensorRT10.8", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	return 0;
}