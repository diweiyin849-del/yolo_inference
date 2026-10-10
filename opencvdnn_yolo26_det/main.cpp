#include"yolo26-det.h"
int main(int argc, char** argv) {
	std::string onnxpath = "D:/yolo_models/yolo26n.onnx";
	std::string labelpath = "D:/yolo_models/yolo-cls.txt";  //opencvdnn不支持yolo26端到端模型里面的算子，除非导出为传统格式。
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
		cv::imshow("YOLO26n + opencvdnn4.11", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}