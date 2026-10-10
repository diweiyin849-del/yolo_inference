#include"yolo11-det.h"
void YOLO11Detector::initConfig(std::string onnxpath, std::string textpath) {
	std::ifstream fp(textpath);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			labels.push_back(name);
		}
	}
	fp.close();
	this->net = cv::dnn::readNetFromONNX(onnxpath);
	this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	//this->net.enableFusion(false);
	//this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	//this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

void YOLO11Detector::detect(cv::Mat& frame) {
	int64 start = cv::getTickCount();
	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	float xy_factor = _max / (float)input_w;
	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false);
	this->net.setInput(blob);

	cv::Mat preds = this->net.forward();
	this->output_h = preds.size[1];
	this->output_w = preds.size[2];

	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	cv::Mat dout(this->output_h,this->output_w, CV_32FC1, (float*)preds.ptr());
	cv::Mat det_output = dout.t(); // 8400x84
	//std::cout << "numRows: " << numRows << " numCols:" << numCols << std::endl;
	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, det_output.cols);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 置信度 0～1之间
		if (score > this->conf_t)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = ((cx - 0.5 * ow) * xy_factor);
			int y = ((cy - 0.5 * oh) * xy_factor);
			int width = (ow * xy_factor);
			int height = (oh * xy_factor);
			cv::Rect box(x, y, width, height);
			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];
		cv::Rect box = boxes[index];
		cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2, 8);
		cv::putText(frame, labels[idx], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);
	}
	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
