#include "yolo26-seg.h"

void YOLO26SegDetector::initConfig(std::string onnxpath, std::string textpath) {
	std::ifstream fp(textpath);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			this->labels.push_back(name);
		}
	}
	fp.close();
	ov::Core ie;
	ov::CompiledModel compiled_model = ie.compile_model(onnxpath, "GPU");
	this->infer_request = compiled_model.create_infer_request();
}

void YOLO26SegDetector::detect(cv::Mat& frame) {
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
	float xy_factor = _max / static_cast<float>(this->input_w);
	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(), true, false);
	memcpy(input_tensor.data<float>(), blob.ptr<float>(), data_s);

	// 推理与返回结果
	this->infer_request.infer();

	ov::Tensor output_tensor = this->infer_request.get_tensor("output0");
	ov::Shape outputDims = output_tensor.get_shape();
	this->output_h = outputDims[1];
	this->output_w = outputDims[2];
	// 后处理,300x38
	cv::Mat dout(this->output_h, this->output_w, CV_32FC1, (float*)output_tensor.data<float>());

	ov::Tensor output_mask = this->infer_request.get_tensor("output1");
	auto oDims1 = output_mask.get_shape();
	this->output_mask_weight = oDims1[1];
	this->output_mask_h = oDims1[2];
	this->output_mask_w = oDims1[3];
	cv::Mat mask1(this->output_mask_weight, this->output_mask_h * this->output_mask_w, CV_32FC1, (float*)output_mask.data<float>());

	// 后处理, 300x38
	cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());
	for (int i = 0; i < dout.rows; i++) {
		//detect
		float score = dout.at<float>(i, 4);
		// 置信度 0～1之间
		if (score > this->score_threshold)
		{
			float x1 = dout.at<float>(i, 0);
			float y1 = dout.at<float>(i, 1);
			float x2 = dout.at<float>(i, 2);
			float y2 = dout.at<float>(i, 3);
			int rx1 = static_cast<int>(x1 * xy_factor);
			int ry1 = static_cast<int>(y1 * xy_factor);
			int rx2 = static_cast<int>(x2 * xy_factor);
			int ry2 = static_cast<int>(y2 * xy_factor);
			int class_id = static_cast<int>(dout.at<float>(i, 5));
			cv::Rect box(rx1, ry1, rx2 - rx1, ry2 - ry1);
			rx1 = std::max(0, box.x);
			ry1 = std::max(0, box.y);
			rx2 = std::min(frame.cols, box.br().x);
			ry2 = std::min(frame.rows, box.br().y);
			cv::Mat m2 = dout.row(i).colRange(output_w - 32, output_w); // 32个掩码系数 [1x32]
			cv::Mat m = m2 * mask1;    //M2:1*32 mask1: 32*25600 M:1*25600 矩阵乘法[1x32] × [32x25600] = [1x25600]
			cv::exp(-m, m);          // m = exp(-m)    矩阵运算代替for遍历;
			cv::add(m, 1.0, m);      // m = m + 1
			cv::divide(1.0, m, m);   // m = 1 / m (sigmoid)
			cv::Mat m1 = m.reshape(1, 160);  //m1:160*160;
			int mx1 = std::max(0, int(rx1 * sx));
			int mx2 = std::max(0, int(rx2 * sx));
			int my1 = std::min(160, int(ry1 * sy));
			int my2 = std::min(160, int(ry2 * sy));
			cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2)); //mask_roi：从完整的 160×160 掩码中切出该检测框对应的矩形区域。此时 mask_roi 的尺寸约为 (y2-y1) × (x2-x1)，值仍在 [0,1]。
			cv::Mat rm, det_mask;
			cv::resize(mask_roi, rm, cv::Size(rx2 - rx1, ry2 - ry1));
			cv::threshold(rm, rm, 0.5, 1.0, cv::THRESH_BINARY);  //二值化代替for遍历;
			rm.convertTo(det_mask, CV_8UC1);
			cv::Mat mask = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
			det_mask.copyTo(mask(cv::Range(ry1, ry2), cv::Range(rx1, rx2))); //拷贝到空白mask,此时mask不是全0了
			add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask); //mask 作为单通道控制信号，与多通道 rgb_mask
			cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1, 8);
			cv::rectangle(frame, cv::Point(box.tl().x, box.tl().y - 20), cv::Point(box.br().x, box.tl().y), cv::Scalar(0, 255, 255), -1);
			cv::putText(frame, this->labels[class_id], cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}

	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::addWeighted(frame, 0.6, rgb_mask, 0.4, 0, frame);
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

}
