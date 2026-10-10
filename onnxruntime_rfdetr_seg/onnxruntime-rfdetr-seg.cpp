#include "rfdetr-seg.h"

double YOLODetector::sigmoid_stable(double x) {
	if (x >= 0.0) {
		return 1.0 / (1.0 + std::exp(-x));
	}
	else {
		return std::exp(x) / (1.0 + std::exp(x));
	}
}

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

	this->session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

	// 3. 使用 GPU（可切换 CPU）
	std::cout << "Using GPU device 0" << std::endl;
	OrtStatusPtr status = OrtSessionOptionsAppendExecutionProvider_CUDA(this->session_options, 0); //为 session_options 添加 CUDA 执行提供者（即使用 GPU 推理）
	// 4. 加载模型
	std::wstring wpath(onnxpath.begin(), onnxpath.end());
	this->session = Ort::Session(this->env, wpath.c_str(), this->session_options);

	// 5. 获取输入输出节点名称（假设各只有一个）
	Ort::AllocatorWithDefaultOptions allocator;

	Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
	this->input_node_names.push_back(input_name_ptr.get());
	Ort::AllocatedStringPtr output_det_ptr = session.GetOutputNameAllocated(0, allocator); //这里的0/1是index
	this->output_node_names.push_back(output_det_ptr.get());
	Ort::AllocatedStringPtr output_label_ptr = session.GetOutputNameAllocated(1, allocator);
	this->output_node_names.push_back(output_label_ptr.get());
	Ort::AllocatedStringPtr output_mask_ptr = session.GetOutputNameAllocated(2, allocator);
	this->output_node_names.push_back(output_mask_ptr.get());

	this->inputNames = { this->input_node_names[0].c_str() }; //准备输入名称数组
	this->outputNames = { this->output_node_names[0].c_str(),this->output_node_names[1].c_str(),this->output_node_names[2].c_str()};//准备输出名称数组
	// 可选：打印输入输出形状
	std::vector<int64_t> input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "Input shape: " << input_shape[1] << "x" << input_shape[2] << "x" << input_shape[3] << std::endl;
	std::vector<int64_t> output_det_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "Output_det shape: " << output_det_shape[1] << "x" << output_det_shape[2] << std::endl;
	std::vector<int64_t> output_label_shape = session.GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "Output_label shape: " << output_label_shape[1] << "x" << output_label_shape[2] << std::endl;
	std::vector<int64_t> output_mask_shape = session.GetOutputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "output_maskput shape: " << output_mask_shape[1] << "x" << output_mask_shape[2] << "x" << output_mask_shape[3] << std::endl;
	this->output_det_h = output_det_shape[1];
	this->output_det_w = output_det_shape[2];
	this->output_label_h = output_label_shape[1];
	this->output_label_w = output_label_shape[2];
	this->output_mask_weight = output_mask_shape[1];
	this->output_mask_h = output_mask_shape[2];
	this->output_mask_w = output_mask_shape[3];
}


void YOLODetector::detect(cv::Mat& frame) {
	int64 start = cv::getTickCount();
	cv::RNG rng;
	// -------------------- 预处理 --------------------
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	// letterbox: 创建正方形画布，填充黑色
	cv::Mat canvas = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(canvas(roi));
	// blobFromImage: 缩放 + 归一化(1/255) + 转为NCHW + BGR->RGB
	float sx = 108.0 / _max;
	float sy = 108.0 / _max;

	cv::Mat resizedimage;
	cv::resize(canvas, resizedimage, cv::Size(432, 432));
	resizedimage.convertTo(resizedimage, CV_32F, 1.0 / 255.0);

	cv::Mat meanImg = cv::Mat::zeros(resizedimage.size(), resizedimage.type());
	meanImg.setTo(cv::Scalar(0.485, 0.456, 0.406));

	cv::Mat stdImg = cv::Mat::zeros(resizedimage.size(), resizedimage.type());
	stdImg.setTo(cv::Scalar(0.229, 0.224, 0.225));

	cv::subtract(resizedimage, meanImg, resizedimage);
	cv::divide(resizedimage, stdImg, resizedimage);

	cv::Mat tensor = cv::dnn::blobFromImage(resizedimage);

	size_t tpixels = 3 * this->input_h * this->input_w;
	std::array<int64_t, 4>input_shape_info{ 1,3,this->input_h,this->input_w };
	Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); //创建 CPU 内存信息对象，用于告诉 ONNX Runtime 张量数据存储在 CPU 的哪个区域。
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, tensor.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
	std::vector<Ort::Value> outputs;
	try {
		outputs = this->session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, inputNames.size(), outputNames.data(), outputNames.size());
	}
	catch (const std::exception& e) {
		std::cerr << "Inference failed: " << e.what() << std::endl;
		return;
	}

	
	// -------------------- 后处理 --------------------
	cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());
	const float* raw_output1 = outputs[0].GetTensorMutableData<float>();
	cv::Mat det_output(this->output_det_h, this->output_det_w, CV_32F, (float*)raw_output1);

	const float* raw_output2 = outputs[1].GetTensorMutableData<float>();
	cv::Mat label_output(this->output_label_h, this->output_label_w, CV_32F, (float*)raw_output2);
	
	const float* raw_output3 = outputs[2].GetTensorMutableData<float>();
	
	for (int i = 0; i < det_output.rows; i++) {
		//最后一个类别是背景;
		cv::Mat classes_scores = label_output.row(i).colRange(0, label_output.cols);
		cv::Point classIdPoint;
		double maxscore;
		minMaxLoc(classes_scores, 0, &maxscore, 0, &classIdPoint);
		maxscore = this->sigmoid_stable(maxscore);
		if (maxscore > this->conf_t)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int box_x = ((cx - 0.5 * ow) * _max);
			int box_y = ((cy - 0.5 * oh) * _max);
			int box_w = (ow * _max);
			int box_h = (oh * _max);
			cv::Rect box(box_x, box_y, box_w, box_h); // 3. 计算画布上的框_max 是画布边长
			int x1 = std::max(0, box.x);
			int y1 = std::max(0, box.y);
			int x2 = std::min(frame.cols, box.br().x);
			int y2 = std::min(frame.rows, box.br().y);
			if (x2 <= x1 || y2 <= y1) continue; // ← 必须加这一句
			cv::Mat m(this->output_mask_h, this->output_mask_w, CV_32F, (float*)(raw_output3 + i * this->output_mask_h * this->output_mask_w));
			cv::exp(-m, m);          // m = exp(-m)    矩阵运算代替for遍历;
			cv::add(m, 1.0, m);      // m = m + 1
			cv::divide(1.0, m, m);   // m = 1 / m (sigmoid)
			// 5. 将画布坐标映射到掩码空间（78x78）
			int mx1 = std::max(0, int(x1 * sx));
			int my1 = std::max(0, int(y1 * sy));
			int mx2 = std::min(108, int(x2 * sx));
			int my2 = std::min(108, int(y2 * sy));
			if (mx2 <= mx1 || my2 <= my1) continue;  // ← 必须加这一句
			cv::Mat mask_roi = m(cv::Range(my1, my2), cv::Range(mx1, mx2)); //mask_roi：从完整的78x78 掩码中切出该检测框对应的矩形区域。此时 mask_roi 的尺寸约为 (y2-y1) × (x2-x1)，值仍在 [0,1]。
			if (mask_roi.empty()) continue;          // ← 加这一句
			cv::Mat rm, det_mask;
			if (x2 - x1 <= 0 || y2 - y1 <= 0) continue;  // ← 加这一句
			cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));
			cv::threshold(rm, rm, 0.5, 1.0, cv::THRESH_BINARY);  //二值化代替for遍历;
			rm.convertTo(det_mask, CV_8UC1);
			cv::Mat mask = cv::Mat::zeros(cv::Size(frame.cols,frame.rows), CV_8UC1);
			det_mask.copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
			// 10. 取出原图区域（frame 大小）作为最终掩码
			add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);
			cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1, 8);
			cv::rectangle(frame, cv::Point(box.tl().x, box.tl().y - 20), cv::Point(box.br().x, box.tl().y), cv::Scalar(0, 255, 255), -1);
			cv::putText(frame, this->labels[classIdPoint.x], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}

	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::addWeighted(frame, 0.6, rgb_mask, 0.4, 0, frame);
	
	cv::putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}

YOLODetector::~YOLODetector() {
	this->session.release();
	this->env.release();
	this->session_options.release();
	std::cout << "destory instance done" << endl;
}