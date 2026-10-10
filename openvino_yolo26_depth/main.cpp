#include"yolo26-depth.h"
int main(int argc, char** argv) {
	std::string onnxpath = "D:/yolo_models/yolo26n-depth.onnx";
	std::shared_ptr<YOLO11Detector> detector(new YOLO11Detector());
	detector->initConfig(onnxpath);
	cv::VideoCapture capture("D:/yolo_models/QQtest.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("YOLO26n-depth + OpenVINO2026", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}