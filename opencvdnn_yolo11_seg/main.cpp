#include "yolo11-seg.h"
int main(int argc, char** argv) {
	std::string onnxpath = "D:/yolo_models/yolo11n-seg.onnx";
	std::string labelpath = "D:/yolo_models/yolo-cls.txt";
	std::shared_ptr<YOLO11SegDetector> detector(new YOLO11SegDetector());
	detector->initConfig(onnxpath, labelpath);
	cv::VideoCapture capture("D:/yolo_models/play_scoers.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("YOLO11ÊµÀý·Ö¸î + Opencvdnn4.11", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}