#include"yolo11-obb.h"
using namespace std;
void YOLO11OBBDetector::initConfig(std::string onnxpath, std::string textpath) {
	std::ifstream fp(textpath);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			labels.push_back(name);
		}
	}
	fp.close();
	ov::Core ie;
	ov::CompiledModel compiled_model = ie.compile_model(onnxpath, "GPU");
	this->infer_request = compiled_model.create_infer_request();
}

void YOLO11OBBDetector::detect(cv::Mat& frame) {
	int64 start = cv::getTickCount();
	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false);
	float x_factor = _max / static_cast<float>(input_w);
	float y_factor = _max / static_cast<float>(input_h);
	ov::Tensor input_tensor = infer_request.get_input_tensor();
	memcpy(input_tensor.data<float>(), blob.ptr<float>(), data_s);
	// 推理与返回结果
	this->infer_request.infer();
	auto output = this->infer_request.get_output_tensor();
	auto outputDims = output.get_shape();
	this->output_h = outputDims[1];
	this->output_w = outputDims[2];
	// post process
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<float> angles;
	cv::Mat dout(this->output_h, this->output_w, CV_32FC1, (float*)output.data<float>());
	cv::Mat det_output = dout.t(); // 21504x20
	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, this->output_h - 1);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 置信度 0～1之间
		if (score > this->t_score)
		{
			angles.push_back(det_output.at<float>(i, this->output_h - 1)); //最后一列是角度
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = ((cx - 0.5 * ow) * x_factor);
			int y = ((cy - 0.5 * oh) * y_factor);
			int width = (ow * x_factor);
			int height = (oh * y_factor);
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
	std::vector<cv::Point> point_pts;
	std::vector<cv::Point> rrt_pts;
	for (size_t i = 0; i < indexes.size(); i++) {
		point_pts.clear();
		rrt_pts.clear();
		int index = indexes[i];
		int idx = classIds[index];
		cv::Rect box = boxes[index];
		point_pts.push_back(box.tl());
		point_pts.push_back(cv::Point(box.x + box.width, box.y));
		point_pts.push_back(box.br());
		point_pts.push_back(cv::Point(box.x, box.y + box.height));
		getRotatedPts(rrt_pts, -angles[index], point_pts, box);
		for (int r = 0; r < 4; r++) {
			cv::line(frame, rrt_pts[r], rrt_pts[(r + 1) % 4], cv::Scalar(255, 0, 255), 2, 8);
		}
		int cx = box.x + box.width / 2;
		int cy = box.y + box.height / 2;
		cv::putText(frame, labels[idx], cv::Point(cx, cy), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);
	}

	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
//利用矩阵2*3变换得到obb的检测框;
void YOLO11OBBDetector::getRotatedPts(std::vector<cv::Point>& rrts, float angle, std::vector<cv::Point>& point_pts, cv::Rect& box) {
	cv::Mat M = cv::Mat::zeros(Size(3, 2), CV_32FC1);
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