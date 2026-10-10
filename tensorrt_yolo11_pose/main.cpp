# include "yolo11-pose.h"
int main() {
	std::shared_ptr<YOLO11TRTPose> detector(new YOLO11TRTPose());
	detector->initConfig("D:/TensorRT-10.8.0.43/bin/yolo11n-pose-fp16.engine");
	cv::VideoCapture capture("D:/yolo_models/QQtest.mp4");
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame);
		cv::imshow("YOLO11×ËÌ¬ÆÀ¹À + TensorRT10.8", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	return 0;
}