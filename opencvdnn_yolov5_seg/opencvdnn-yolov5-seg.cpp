#include "yolov5-seg.h"

void YOLODetector::initConfig(std::string onnxpath, std::string textpath) {
	// 1. 读取类别文件
	std::ifstream fp(textpath);
	std::string name;
	while (std::getline(fp, name)) {
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


void YOLODetector::detect(cv::Mat& frame) {
	int64 start = cv::getTickCount();
	cv::RNG rng;
	// -------------------- 预处理 --------------------
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	// letterbox: 创建正方形画布，填充黑色
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	float xy_factor = static_cast<float>(_max) / static_cast<float>(this->input_w);
	float sx = 160.0f / _max;
	float sy = 160.0f / _max;
	// blobFromImage: 缩放 + 归一化(1/255) + 转为NCHW + BGR->RGB
	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(), true, false);
	this->net.setInput(blob);
	// 获取所有输出层名称
	std::vector<cv::String> outputNames = net.getUnconnectedOutLayersNames();
	std::vector<cv::Mat> outputs;
	net.forward(outputs, outputNames);

	cv::Mat preds = outputs[0];
	cv::Mat maskOutput = outputs[1];

	// 获取检测头维度
	this->output_h = preds.size[1];  // 例如 300
	this->output_w = preds.size[2];  // 例如 38
	cv::Mat dout(this->output_h, this->output_w, CV_32F, preds.ptr<float>());
	// 获取掩码头维度
	this->output_mask_weight = maskOutput.size[1];  // 例如 32
	this->output_mask_h = maskOutput.size[2];       // 例如 160
	this->output_mask_w = maskOutput.size[3];       // 例如 160
	cv::Mat mask1(this->output_mask_weight, this->output_mask_h * this->output_mask_w, CV_32FC1, (float*)maskOutput.ptr<float>());
	// 后处理, 1x25200x85
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<cv::Mat> masks;
	std::vector<float> confidences;
	cv::Mat det_output(this->output_h, this->output_w, CV_32FC1, (float*)preds.ptr<float>());
	for (int i = 0; i < det_output.rows; i++) {
		float confidence = det_output.at<float>(i, 4);
		if (confidence < 0.25) {
			continue;
		}
		cv::Mat classes_scores = det_output.row(i).colRange(5, output_w - 32);
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
			int x = static_cast<int>((cx - 0.5 * ow) * xy_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * xy_factor);
			int width = static_cast<int>(ow * xy_factor);
			int height = static_cast<int>(oh * xy_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;
			cv::Mat mask2 = det_output.row(i).colRange(output_w - 32, output_w);
			boxes.push_back(box);
			masks.push_back(mask2);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];
		cv::Rect box = boxes[index];
		int x1 = std::max(0, box.x);
		int y1 = std::max(0, box.y);
		int x2 = std::min(frame.cols, box.br().x);
		int y2 = std::min(frame.rows, box.br().y);
		cv::Mat m2 = masks[index];
		cv::Mat m = m2 * mask1;
		//sigmoid函数;
		cv::exp(-m, m);          // m = exp(-m)    矩阵运算代替for遍历;
		cv::add(m, 1.0, m);      // m = m + 1
		cv::divide(1.0, m, m);   // m = 1 / m   (sigmoid)
		cv::Mat m1 = m.reshape(1, 160);
		int mx1 = std::max(0, int(x1 * sx));
		int mx2 = std::max(0, int(x2 * sx));
		int my1 = std::min(160, int(y1 * sy));
		int my2 = std::min(160, int(y2 * sy));
		cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
		cv::Mat rm, det_mask;
		cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));
		cv::threshold(rm, rm, 0.5, 1.0, cv::THRESH_BINARY);  //二值化代替for遍历;
		rm.convertTo(det_mask, CV_8UC1);
		cv::Mat mask = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
		det_mask.copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
		add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);
		cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 1, 8);
		cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20), cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
		cv::putText(frame, labels[idx], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
	}

	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::addWeighted(frame, 0.6, rgb_mask, 0.4, 0, frame);
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
