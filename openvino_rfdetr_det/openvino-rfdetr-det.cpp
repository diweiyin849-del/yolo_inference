#include"rfdetr_det.h"

double YOLO11Detector::sigmoid_stable(double x) {
	if (x >= 0.0) {
		return 1.0 / (1.0 + std::exp(-x));
	}
	else {
		return std::exp(x) / (1.0 + std::exp(x));
	}
}

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
	ov::CompiledModel compiled_model = ie.compile_model(onnxpath, "NPU");
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
	float xy_factor = _max / (float)this->input_w;

	cv::Mat resizedimage;
	cv::resize(image, resizedimage, cv::Size(576, 576));
	resizedimage.convertTo(resizedimage, CV_32F, 1.0 / 255.0);

	cv::Mat meanImg = cv::Mat::zeros(resizedimage.size(), resizedimage.type());
	meanImg.setTo(cv::Scalar(0.485, 0.456, 0.406));

	cv::Mat stdImg = cv::Mat::zeros(resizedimage.size(), resizedimage.type());
	stdImg.setTo(cv::Scalar(0.229, 0.224, 0.225));

	cv::subtract(resizedimage, meanImg, resizedimage);
	cv::divide(resizedimage, stdImg, resizedimage);

	cv::Mat blob = cv::dnn::blobFromImage(resizedimage);

	//memcpy 将blob的数据拷贝到输入张量中。data_s 应该是预先计算好的数据大小（3 * input_h * input_w * sizeof(float)）。
	memcpy(input_tensor.data<float>(), blob.ptr<float>(), this->data_s);

	// 推理与返回结果
	this->infer_request.infer(); //同步执行推理，模型计算完成后返回。
	ov::Tensor dets_tensor = infer_request.get_tensor("dets");      // 形状: {1, 300, 4}
	ov::Tensor labels_tensor = infer_request.get_tensor("labels");  // 形状: {1, 300, 91}

	// 获取形状
	ov::Shape det_shape = dets_tensor.get_shape();   // {1, 300, 4}
	ov::Shape label_shape = labels_tensor.get_shape(); // {1, 300, 91}

	this->output_det_h = det_shape[1];   // 300
	this->output_det_w = det_shape[2];   // 4
	this->output_label_h= label_shape[1]; // 300
	this->output_label_w = label_shape[2]; // 91
	std::cout << "output_det_h: " << this->output_det_h << std::endl;
	std::cout << "output_det_w: " << this->output_det_w << std::endl;
	std::cout << "output_label_h: " << this->output_label_h << std::endl;
	std::cout << "output_label_w: " << this->output_label_w << std::endl;

	cv::Mat det_output(this->output_det_h, this->output_det_w, CV_32F, (float*)dets_tensor.data());
	cv::Mat label_output(this->output_label_h, this->output_label_w, CV_32F, (float*)labels_tensor.data());

	for (int i = 0; i < det_output.rows; i++) {
		//最后一个类别是背景;
		cv::Mat classes_scores = label_output.row(i).colRange(0, label_output.cols);
		cv::Point classIdPoint;
		double maxscore;
		minMaxLoc(classes_scores, 0, &maxscore, 0, &classIdPoint);
		maxscore = this->sigmoid_stable(maxscore);
		// 置信度 0～1之间
		if (maxscore > this->conf_t)
		{   
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = ((cx - 0.5 * ow) * _max); //输出是0-1归一化坐标，区别于YOLO
			int y = ((cy - 0.5 * oh) * _max);
			int width = (ow * _max);
			int height = (oh * _max);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;
			cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1, 8);
			cv::rectangle(frame, cv::Point(box.tl().x, box.tl().y - 20), cv::Point(box.br().x, box.tl().y), cv::Scalar(0, 255, 255), -1);
			cv::putText(frame, this->labels[classIdPoint.x], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}

	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
