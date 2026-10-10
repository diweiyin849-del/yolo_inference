#include "openvino-async-detect.h"
int main(int argc, char** argv) {
	std::string onnxpath = "D:/yolo_models/yolo11n.onnx";
	std::string labelpath = "D:/yolo_models/yolo-cls.txt";
	std::shared_ptr<YOLO11AsyncDetector> detector(new YOLO11AsyncDetector());
	detector->initConfig(onnxpath, labelpath);
	detector->detect("D:/yolo_models/play_scoers.mp4");

	return 0;
}