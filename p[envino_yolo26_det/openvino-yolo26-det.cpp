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
	ov::Core ie;
	ov::CompiledModel compiled_model = ie.compile_model(onnxpath, "GPU");
	this->infer_request = compiled_model.create_infer_request();
}

void YOLO11Detector::detect(cv::Mat& frame) {
	// 获取输入格式
	int64 start = cv::getTickCount();
	// infer_request 内部已经有一个输入缓冲区（Tensor）
	// get_input_tensor() 只是把这个缓冲区暴露给你，让你往里面写数据
	ov::Tensor input_tensor = infer_request.get_input_tensor(); //get_input_tensor() 返回模型的输入张量，用于后续填充数据。

	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	float xy_factor = _max / (float)input_w;

	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false);
	//memcpy 将blob的数据拷贝到输入张量中。data_s 应该是预先计算好的数据大小（3 * input_h * input_w * sizeof(float)）。
	memcpy(input_tensor.data<float>(), blob.ptr<float>(), data_s);

	// 推理与返回结果
	this->infer_request.infer(); //同步执行推理，模型计算完成后返回。
	ov::Tensor output = this->infer_request.get_output_tensor();
	ov::Shape outputDims = output.get_shape();
	this->output_h = outputDims[1]; // 300
	this->output_w = outputDims[2]; // 6
	cv::Mat det_output(this->output_h, this->output_w, CV_32F, (float*)output.data());
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
