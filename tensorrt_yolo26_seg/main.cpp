#include "yolo26-seg.h"
std::string label_map = "D:/yolo_models/yolo-cls.txt";
int main(int argc, char** argv) {
	std::string enginefile = "D:/TensorRT-10.8.0.43/bin/yolo26n-seg-fp16.engine";
	cv::VideoCapture cap("D:/yolo_models/play_scoers.mp4");
	cv::Mat frame;
	auto detector = std::make_unique<YOLOv26TRTSegment>();
	detector->initConfig(enginefile, label_map);
	while (true) {
		cap.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("YOLO26n-seg + TensorRT10.8 ÊµÀý·Ö¸îÑÝÊ¾", frame);
		char c = cv::waitKey(1);
		if (c == 27) { 
			break;
		}
	}
	return 0;
}