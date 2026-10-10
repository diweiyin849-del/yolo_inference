#include"yolo26-det.h"

void YOLO11Detector::initConfig(std::string onnxpath, std::string textpath) {
	std::ifstream fp(textpath);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			this->labels.push_back(name);
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
	float xy_factor = _max / (float)this->input_w;

	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(), true, false);
	this->net.setInput(blob);
	cv::Mat preds = this->net.forward();
	this->output_h = preds.size[1];
	this->output_w = preds.size[2];
	cv::Mat det_output(this->output_h, this->output_w, CV_32F, (float*)preds.ptr());
	for (int i = 0; i < det_output.rows; i++) {
		float score = det_output.at<float>(i, 4);
		// 置信度 0～1之间
		if (score > this->conf_t)
		{
			float x1 = det_output.at<float>(i, 0);
			float y1 = det_output.at<float>(i, 1);
			float x2 = det_output.at<float>(i, 2);
			float y2 = det_output.at<float>(i, 3);
			int rx1 = static_cast<int>(x1 * xy_factor);
			int ry1 = static_cast<int>(y1 * xy_factor);
			int rx2 = static_cast<int>(x2 * xy_factor);
			int ry2 = static_cast<int>(y2 * xy_factor);
			int class_id = static_cast<int>(det_output.at<float>(i, 5));
			cv::Rect box(rx1, ry1, rx2 - rx1, ry2 - ry1);
			cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1, 8);
			cv::rectangle(frame, cv::Point(box.tl().x, box.tl().y - 20), cv::Point(box.br().x, box.tl().y), cv::Scalar(0, 255, 255), -1);
			cv::putText(frame, this->labels[class_id], cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}

	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
