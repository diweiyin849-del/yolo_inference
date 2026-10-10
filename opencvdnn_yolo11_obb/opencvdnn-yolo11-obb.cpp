#include "yolo11-obb.h"

void YOLO11ObbDetector::initConfig(std::string onnxpath) {
	this->net = cv::dnn::readNetFromONNX(onnxpath);
	//this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	//this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	this->net.enableFusion(false);
	this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

void YOLO11ObbDetector::detect(cv::Mat& frame, std::vector<std::string>& classNames) {
	int64 start = cv::getTickCount();
	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));

	float x_factor = image.cols / static_cast<float>(this->input_w);
	float y_factor = image.rows / static_cast<float>(this->input_h);

	// 推理
	cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	cv::Mat preds = this->net.forward();

	// post process
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<float> angles;
	this->output_h = preds.size[1];
	this->output_w = preds.size[2];
	cv::Mat dout(this->output_h,this->output_w, CV_32F, preds.ptr<float>());
	cv::Mat det_output = dout.t(); // 8400x20
	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, det_output.cols - 1);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 置信度 0～1之间
		if (score > this->score_threshold)
		{
			angles.push_back(det_output.at<float>(i, det_output.cols - 1));
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;
			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);

	std::vector<cv::Point> box_pts;
	std::vector<cv::Point> rrt_pts;
	for (size_t i = 0; i < indexes.size(); i++) {
		box_pts.clear();
		rrt_pts.clear();
		int index = indexes[i];
		int idx = classIds[index];
		cv::Rect box = boxes[index];
		box_pts.push_back(box.tl());
		box_pts.push_back(cv::Point(box.x + box.width, box.y));
		box_pts.push_back(box.br());
		box_pts.push_back(cv::Point(box.x, box.y + box.height));
		getRotatedPts(rrt_pts, -angles[index], box_pts, box);
		for (int r = 0; r < 4; r++) {
			cv::line(frame, rrt_pts[r], rrt_pts[(r + 1) % 4], cv::Scalar(255, 0, 255), 2, 8);
		}
		int cx = box.x + box.width / 2;
		int cy = box.y + box.height / 2;
		cv::putText(frame, classNames[idx], cv::Point(cx, cy), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 255, 0));
	}
	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}

//利用矩阵2*3变换得到obb的检测框;
void YOLO11ObbDetector::getRotatedPts(std::vector<cv::Point>& rrts, float angle, std::vector<cv::Point>& point_pts, cv::Rect& box) {
	cv::Mat M = cv::Mat::zeros(cv::Size(3, 2), CV_32FC1);
	double alpha = cos(angle);   //cos   sin   (1 - cos) * center_x - sin * center_y
	double beta = sin(angle);   // -sin  cos   sin * center_x + (1 - cos) * center_y
	int center_x = box.x + box.width / 2;
	int center_y = box.y + box.height / 2;
	double tx = (1 - alpha) * center_x - beta * center_y;
	double ty = beta * center_x + (1 - alpha) * center_y;
	M.at<float>(0, 0) = alpha;
	M.at<float>(1, 1) = alpha;
	M.at<float>(0, 1) = beta;
	M.at<float>(1, 0) = -beta;
	M.at<float>(0, 2) = tx;
	M.at<float>(1, 2) = ty;
	for (cv::Point pt : point_pts) {
		float x0 = pt.x;
		float y0 = pt.y;
		float x = M.at<float>(0, 0) * x0 + M.at<float>(0, 1) * y0 + M.at<float>(0, 2); //本质矩阵线性变换;
		float y = M.at<float>(1, 0) * x0 + M.at<float>(1, 1) * y0 + M.at<float>(1, 2); //本质矩阵线性变换;
		rrts.push_back(cv::Point(x, y));
	}
}