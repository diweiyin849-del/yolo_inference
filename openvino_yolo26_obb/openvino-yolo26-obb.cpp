#include"yolo26-obb.h"
void YOLO26OBBDetector::initConfig(std::string onnxpath, std::string textpath) {
	std::ifstream fp(textpath);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			labels.push_back(name);
		}
	}
	fp.close();
	ov::Core ie;
	ov::CompiledModel compiled_model = ie.compile_model(onnxpath, "CPU");
	this->infer_request = compiled_model.create_infer_request();
}

void YOLO26OBBDetector::detect(cv::Mat& frame) {
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

	ov::Tensor output = this->infer_request.get_output_tensor();

	ov::Shape outputDims = output.get_shape();
	this->output_h = outputDims[1];
	this->output_w = outputDims[2];

	//300*7
	cv::Mat dout(this->output_h, this->output_w, CV_32FC1, (float*)output.data<float>());
	// 后处理, 1x300x7 [x, y, w, h, conf, classid, angle]
	std::vector<cv::Point> point_pts;
	std::vector<cv::Point> rrt_pts;
	for (int i = 0; i < dout.rows; i++) {
		float score = dout.at<float>(i, 4);
		// 置信度 0～1之间
		if (score > this->t_score)
		{
			point_pts.clear();
			rrt_pts.clear();
			float cx = dout.at<float>(i, 0);
			float cy = dout.at<float>(i, 1);
			float ow = dout.at<float>(i, 2);
			float oh = dout.at<float>(i, 3);
			int x = ((cx - 0.5 * ow) * x_factor);
			int y = ((cy - 0.5 * oh) * y_factor);
			int width = (ow * x_factor);
			int height = (oh * y_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;
			int class_id = static_cast<int>(dout.at<float>(i, 5));
			float angle = dout.at<float>(i, this->output_w - 1);
			point_pts.push_back(box.tl());
			point_pts.push_back(cv::Point(box.x + box.width, box.y));
			point_pts.push_back(box.br());
			point_pts.push_back(cv::Point(box.x, box.y + box.height));
			getRotatedPts(rrt_pts, -angle, point_pts, box);
			for (int r = 0; r < 4; r++) {
				cv::line(frame, rrt_pts[r], rrt_pts[(r + 1) % 4], cv::Scalar(255, 0, 255), 2, 8);
			}
			int center_x = box.x + box.width / 2;
			int center_y = box.y + box.height / 2;
			cv::putText(frame, this->labels[class_id], cv::Point(center_x, center_y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);
		}
	}


	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
//利用矩阵2*3变换得到obb的检测框;
void YOLO26OBBDetector::getRotatedPts(std::vector<cv::Point>& rrts, float angle, std::vector<cv::Point>& point_pts, cv::Rect& box) {
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