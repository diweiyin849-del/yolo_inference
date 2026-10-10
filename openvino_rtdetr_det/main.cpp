#include"rtdetr_det.h"
int main(int argc, char** argv) {

	std::string onnxpath = "D:/yolo_models/rtdetr-l.onnx";
	std::string labelpath = "D:/yolo_models/yolo-cls.txt";
	std::shared_ptr<YOLO11Detector> detector(new YOLO11Detector());
	detector->initConfig(onnxpath, labelpath);
	cv::VideoCapture capture("D:/yolo_models/play_scoers.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("RTdetr + OpenVINO2026", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}