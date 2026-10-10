#include"yolo26-depth.h"
using namespace std;
int main(int argc, char** argv) {
	std::unique_ptr<YOLOv81112TRTDetector> detector = std::make_unique<YOLOv81112TRTDetector>();
	detector->initConfig("D:/yolo_models/yolo26n-depth.engine");
	cv::VideoCapture capture("D:/yolo_models/QQtest.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("YOLO26n-depth + TensorRT10.8", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	return 0;
}