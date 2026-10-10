#include "yolo11-pose.h"
using namespace std;
int main(int argc, char** argv) {
	std::string onnxpath = "D:/yolo_models/yolo11n-pose.onnx";
	std::unique_ptr<YOLO11PoseDetector> detector = make_unique<YOLO11PoseDetector>();
	detector->initConfig(onnxpath);
	cv::VideoCapture capture("D:/yolo_models/QQtest.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("YOLO11×ËÌ¬ÆÀ¹À + Opencvdnn4.11", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}