#include"rtdetr-det.h"
using namespace std;
std::string label_map = "D:/yolo_models/yolo-cls.txt";
int main(int argc, char** argv) {
	std::unique_ptr<YOLOv81112TRTDetector> detector = std::make_unique<YOLOv81112TRTDetector>();
	detector->initConfig("D:/yolo_models/rtdetr-l.engine", label_map);
	cv::VideoCapture capture("D:/yolo_models/play_scoers.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("RTdetr + TensorRT10.8", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	return 0;
}