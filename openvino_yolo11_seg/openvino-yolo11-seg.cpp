#include "yolo11-seg.h"
void YOLO11SegDetector::initConfig(std::string onnxpath, std::string textpath) {
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

void YOLO11SegDetector::detect(cv::Mat& frame) {
	std::vector<DetectResult> results;
	// 获取输入格式
	int64 start = cv::getTickCount();
	// 请求网络输入
	ov::Tensor input_tensor = infer_request.get_input_tensor();
	cv::RNG rng(12345);
	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	float sx = 160.0f / static_cast<float>(_max);
	float sy = 160.0f / static_cast<float>(_max);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	float xy_factor = _max / static_cast<float>(input_w);
	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false);
	memcpy(input_tensor.data<float>(), blob.ptr<float>(), data_s);

	// 推理与返回结果
	this->infer_request.infer();

	ov::Tensor output0 = this->infer_request.get_tensor("output0");
	auto outputDims = output0.get_shape();
	this->output_h = outputDims[1];
	this->output_w = outputDims[2];
	// 后处理,116*8400
	cv::Mat dout(this->output_h, this->output_w, CV_32FC1, (float*)output0.data<float>());
	cv::Mat det_output = dout.t();

	ov::Tensor output1 = this->infer_request.get_tensor("output1");
	auto oDims1 = output1.get_shape();
	this->output_mask_weight = oDims1[1];
	this->output_mask_h = oDims1[2];
	this->output_mask_w = oDims1[3];
	cv::Mat mask1(this->output_mask_weight, this->output_mask_h * this->output_mask_w, CV_32FC1, (float*)output1.data<float>());

	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Mat> masks;
	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, this->output_h - 32);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 置信度 0～1之间
		if (score > 0.25)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = ((cx - 0.5 * ow) * xy_factor);//出来是相对于640*640像素坐标，转为相对于image的坐标;
			int y = ((cy - 0.5 * oh) * xy_factor);
			int width = (ow * xy_factor);
			int height = (oh * xy_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;
			cv::Mat mask2 = det_output.row(i).colRange(this->output_h - 32, this->output_h);
			masks.push_back(mask2);
			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// 1. NMS过滤 → 2. 掩码计算 → 3. 坐标映射 → 4. 二值化
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());
	for (size_t i = 0; i < indexes.size(); i++) {
		DetectResult dr;
		int index = indexes[i];
		int idx = classIds[index];
		dr.box = boxes[index];
		dr.classId = idx;
		dr.conf = confidences[index];
		cv::Rect box = boxes[index];
		int x1 = std::max(0, box.x);
		int y1 = std::max(0, box.y);
		int x2 = std::min(frame.cols, box.br().x);
		int y2 = std::min(frame.rows, box.br().y);
		cv::Mat m2 = masks[index]; // 32个掩码系数 [1x32]
		cv::Mat m = m2 * mask1;    //M2:1*32 mask1: 32*25600 M:1*25600 矩阵乘法[1x32] × [32x25600] = [1x25600]
		cv::exp(-m, m);          // m = exp(-m)    矩阵运算代替for遍历;
		cv::add(m, 1.0, m);      // m = m + 1
		cv::divide(1.0, m, m);   // m = 1 / m   (sigmoid)
		cv::Mat m1 = m.reshape(1, 160);  //m1:160*160;
		int mx1 = std::max(0, static_cast<int>(x1 * sx));
		int mx2 = std::max(0, static_cast<int>(x2 * sx));
		int my1 = std::min(160, static_cast<int>(y1 * sy));
		int my2 = std::min(160, static_cast<int>(y2 * sy));
		cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2)); //mask_roi：从完整的 160×160 掩码中切出该检测框对应的矩形区域。此时 mask_roi 的尺寸约为 (y2-y1) × (x2-x1)，值仍在 [0,1]。
		cv::Mat rm, det_mask;
		cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));
		cv::threshold(rm, rm, 0.5, 1.0, cv::THRESH_BINARY);  //二值化代替for遍历;
		rm.convertTo(det_mask, CV_8UC1); //32FC1->8UC1
		cv::Mat mask = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
		det_mask.copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2))); //拷贝到空白mask
		add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);
		cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 1, 8);
		cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20), cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
		results.push_back(dr);
	}

	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::addWeighted(frame, 0.6, rgb_mask, 0.4, 0, frame);
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
	for (DetectResult dr : results) {
		cv::Rect box = dr.box;
		cv::putText(frame, labels[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
}
