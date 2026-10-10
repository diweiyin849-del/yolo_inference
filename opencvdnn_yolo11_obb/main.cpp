#include "yolo11-obb.h"

std::string label_map = "D:/yolo_models/dotav1.txt";
int main(int argc, char** argv) {
	std::vector<std::string> classNames;
	std::ifstream fp(label_map);
	std::string name;
	while (!fp.eof()) {
		getline(fp, name);
		if (name.length()) {
			classNames.push_back(name);
		}
	}
	fp.close();
	std::shared_ptr<YOLO11ObbDetector> detector(new YOLO11ObbDetector());
	detector->initConfig("D:/yolo_models/yolo11n-obb.onnx");
	cv::VideoCapture capture("D:/yolo_models/test_ship.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) break;
		detector->detect(frame,classNames);
		cv::imshow("YOLO11-obb + opencvdnn 4.11¼ì²â", frame);
		if (cv::waitKey(1) == 27) break;
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
