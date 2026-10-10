#include "yolo11-seg.h"
std::string label_map = "D:/yolo_models/yolo-cls.txt";
int main(int argc, char** argv) {
	std::vector<string> classNames;
	std::ifstream fp(label_map);
	std::string name;
	std::cout << "CUDA devices: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
	while (getline(fp, name)) {
		if (!name.empty()) {
			classNames.push_back(name);
		}
	}
	fp.close();
	std::string enginefile = "D:/TensorRT-10.8.0.43/bin/yolo11n-seg-fp16.engine";
	cv::VideoCapture cap("D:/yolo_models/play_scoers.mp4");
	cv::Mat frame;
	auto detector = std::make_unique<YOLOv8TRTSegment>();
	detector->initConfig(enginefile, 0.25, 0.25);
	std::vector<DetectResult> results;
	while (true) {
		cap.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame, results);
		for (DetectResult dr : results) {
			cv::Rect box = dr.box;
			cv::putText(frame, classNames[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
		cv::imshow("YOLO11n-seg + TensorRT10.8 ÊµÀý·Ö¸îÑÝÊ¾", frame);
		char c = cv::waitKey(1);
		if (c == 27) { 
			break;
		}
		results.clear();
	}
	return 0;
}