#include"yolo11-det.h"
std::string label_map = "D:/yolo_models/yolo-cls.txt";
int main(int argc, char** argv) {
	std::vector<string> classNames;
	std::ifstream fp(label_map);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			classNames.push_back(name);
		}
	}
	fp.close();
	std::unique_ptr<YOLOv81112TRTDetector> detector = std::make_unique<YOLOv81112TRTDetector>();
	detector->initConfig("D:/TensorRT-10.8.0.43/bin/yolo11n-det-fp16.engine");
	cv::VideoCapture capture("D:/yolo_models/play_scoers.mp4");
	cv::Mat frame;
	std::vector<DetectResult> results;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame, results);
		for (DetectResult dr : results) {
			cv::Rect box = dr.box;
			cv::putText(frame, classNames[dr.classId], cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
		cv::imshow("YOLO11n∂‘œÛºÏ≤‚ + TensorRT10.8", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
		results.clear();
	}
	return 0;
}