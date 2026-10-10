#include"rfdetr-det.h"
using namespace std;
std::string label_map = "D:/yolo_models/pill.txt";
int main(int argc, char** argv) {
	std::unique_ptr<YOLOv81112TRTDetector> detector = std::make_unique<YOLOv81112TRTDetector>();
	detector->initConfig("D:/yolo_models/rfdetr-medium-pill.engine", label_map);
	cv::VideoCapture capture("C:/Users/24150/Desktop/Frame_404.jpg");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("RFdetr + TensorRT10.8", frame);
		char c = cv::waitKey(0);
		if (c == 27) {
			break;
		}
	}
	return 0;
}