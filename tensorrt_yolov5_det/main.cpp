#include "yolov5-det.h"
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
	std::string enginefile = "D:/TensorRT-10.8.0.43/bin/yolov5s-fp16.engine";
	cv::VideoCapture cap("D:/yolo_models/play_scoers.mp4");
	cv::Mat frame;
	auto detector = std::make_shared<YOLOv5TRTDetector>();
	detector->initConfig(enginefile);
	std::vector<DetectResult> results;
	while (true) {
		cap.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame, results);
		for (DetectResult dr : results) {
			cv::Rect box = dr.box;
			cv::putText(frame, classNames[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
		}
		cv::imshow("YOLOv5s + TensorRT10.8бнЪО", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
		results.clear();
	}
	return 0;
}