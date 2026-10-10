#include "yolov5-seg.h"
std::string labels_txt_file = "D:/yolo_models/yolo-cls.txt";
std::vector<std::string> readClassNames();
std::vector<std::string> readClassNames()
{
	std::vector<std::string> classNames;

	std::ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}

int main(int argc, char** argv) {
	std::vector<std::string> labels = readClassNames();
	std::string enginefile = "D:/TensorRT-10.8.0.43/bin/yolov5s-seg-fp16.engine";
	cv::VideoCapture cap("D:/yolo_models/QQtest.mp4");
	cv::Mat frame;
	auto detector = std::make_shared<YOLOv5TRTSegment>();
	detector->initConfig(enginefile);
	std::vector<DetectResult> results;
	while (true) {
		bool ret = cap.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame, results);
		for (DetectResult dr : results) {
			cv::Rect box = dr.box;
			cv::putText(frame, labels[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
		}
		cv::imshow("YOLOv5s-7.0 + TensorRT10.8", frame);
		char c = cv::waitKey(1);
		if (c == 27) { 
			break;
		}
		results.clear();
	}
	return 0;
}