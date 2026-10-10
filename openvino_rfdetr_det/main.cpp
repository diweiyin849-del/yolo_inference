#include"rfdetr_det.h"
int main(int argc, char** argv) {

	std::string onnxpath = "D:/yolo_models/rfdetr-medium-pill.onnx";
	std::string labelpath = "D:/yolo_models/pill.txt";
	std::shared_ptr<YOLO11Detector> detector(new YOLO11Detector());
	detector->initConfig(onnxpath, labelpath);
	cv::VideoCapture capture("C:/Users/24150/Desktop/Frame_404.jpg");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("RFdetr + OpenVINO2026", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}