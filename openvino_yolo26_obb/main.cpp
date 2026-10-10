#include"yolo26-obb.h"
using namespace std;
int main(int argc, char** argv) {
	std::string onnxpath = "D:/yolo_models/yolo26n-obb.onnx";
	std::string labelpath = "D:/yolo_models/dotav1.txt";
	std::unique_ptr< YOLO26OBBDetector> detector = make_unique< YOLO26OBBDetector>();
	detector->initConfig(onnxpath, labelpath);
	cv::VideoCapture capture("D:/yolo_models/test_ship.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("YOLO26-obb + OpenVINO2026", frame);
		char c = cv::waitKey(1);
		if (c == 27) { 
			break;
		}
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}