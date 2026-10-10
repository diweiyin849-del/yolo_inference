#include "yolo26-pose.h"
int main(int argc, char** argv) {
	std::string onnxpath = "D:/yolo_models/yolo26n-pose.onnx";
	std::unique_ptr<YOLO26PoseDetector> detector = make_unique<YOLO26PoseDetector>();
	detector->initConfig(onnxpath);
	cv::VideoCapture capture("D:/yolo_models/QQtest.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("YOLO26n-pose + OpenVINO2026", frame);
		char c = cv::waitKey(1);
		if (c == 27) { 
			break;
		}
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}