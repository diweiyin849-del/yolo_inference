#include "yolo11n-pose.h"

void YOLODetector::initConfig(std::string onnxpath) {
	cv::RNG rng;
	for (int i = 0; i < 17; i++) {
		int a = rng.uniform(0, 255);
		int b = rng.uniform(0, 255);
		int c = rng.uniform(0, 255);
		this->color_tables.push_back(cv::Scalar(a, b, c));
	}
	this->session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
	// 3. 使用 GPU（可切换 CPU）
	std::cout << "Using GPU device 0" << std::endl;
	//OrtStatusPtr status1 = OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);
	//OrtStatusPtr status1 = OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0);
	OrtStatusPtr status = OrtSessionOptionsAppendExecutionProvider_CPU(this->session_options, 0); //为 session_options 添加 CUDA 执行提供者（即使用 GPU 推理）
	std::wstring wpath(onnxpath.begin(), onnxpath.end());
	this->session = Ort::Session(this->env, wpath.c_str(), this->session_options);

	// 5. 获取输入输出节点名称（假设各只有一个）
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator); //输入名称和对应索引
	this->input_node_names.push_back(input_name_ptr.get());
	Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);//输出名称和对应索引
	this->output_node_names.push_back(output_name_ptr.get());

	this->inputNames = { input_node_names[0].c_str() }; //准备输入名称数组,c风格
	this->outputNames = { output_node_names[0].c_str() };//准备输出名称数组,c风格
	// 可选：打印输入输出形状
	std::vector<int64_t> input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "Input shape: " << input_shape[1] << "x" << input_shape[2] << "x" << input_shape[3] << std::endl;
	std::vector<int64_t> output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "Output shape: " << output_shape[1] << "x" << output_shape[2] << std::endl;
	this->output_h = output_shape[1];
	this->output_w = output_shape[2];
}

void YOLODetector::detect(cv::Mat& frame) {
	int64 start = cv::getTickCount();
	// -------------------- 左上角填充预处理 --------------------
	/*
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	// letterbox: 创建正方形画布，填充黑色
	cv::Mat canvas = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(canvas(roi));
	float xy_factor = static_cast<float>(_max) / static_cast<float>(this->input_w);

	cv::Mat m1 = cv::Mat::zeros(cv::Size(3, 17), CV_32FC1);
	for (int i = 0; i < 17; i++) {
		m1.at<float>(i, 0) = xy_factor;
		m1.at<float>(i, 1) = xy_factor;
		m1.at<float>(i, 2) = 1.0f;
	}
	// blobFromImage: 缩放 + 归一化(1/255) + 转为NCHW + BGR->RGB
	cv::Mat tensor = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(), true, false);
	*/

	float scale = std::min(static_cast<float>(this->input_w) / frame.cols, static_cast<float>(this->input_h) / frame.rows);
	int new_w = static_cast<int>(frame.cols * scale);
	int new_h = static_cast<int>(frame.rows * scale);
	cv::Mat resized;
	cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
	cv::Mat canvas = cv::Mat::zeros(cv::Size(this->input_w, this->input_h), CV_8UC3);
	int left = (this->input_w - new_w) / 2;
	int top = (this->input_h - new_h) / 2;
	resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));
	cv::Mat m1 = cv::Mat::zeros(cv::Size(3, 17), CV_32FC1);
	for (int i = 0; i < 17; i++) {
		m1.at<float>(i, 0) = 1.0/scale;
		m1.at<float>(i, 1) = 1.0/scale;
		m1.at<float>(i, 2) = 1.0f;
	}
	cv::Mat tensor = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

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

	const float* raw_output = outputs[0].GetTensorMutableData<float>();

	cv::Mat dout(this->output_h, this->output_w, CV_32F, (float*)raw_output);
	// 后处理, 1x56x8400
	std::vector<cv::Rect> boxes;
	std::vector<cv::Mat> kypts;
	std::vector<float> confidences;
	cv::Mat det_output = dout.t(); // 8400x56
	for (int i = 0; i < det_output.rows; i++) {
		float conf = det_output.at<float>(i, 4);
		// 置信度 0～1之间
		if (conf > this->conf_t)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = ((cx - 0.5 * ow - left) / scale);
			int y = ((cy - 0.5 * oh - top) / scale);
			int width = (ow / scale);
			int height = (oh / scale);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;
			boxes.push_back(box);
			cv::Mat pts = det_output.row(i).colRange(5, this->output_h);
			kypts.push_back(pts);
			confidences.push_back(conf);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (int idx : indexes) {
		cv::rectangle(frame, boxes[idx], cv::Scalar(0, 0, 255), 2, 8);
		cv::putText(frame, "person", boxes[idx].tl(), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
	}
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		cv::Mat one_kypts = kypts[index];
		cv::Mat m2 = one_kypts.reshape(1, 17);// 重塑（reshape）17x3 自动计算列数 
		for (int j = 0; j < 17; j++) {
			m2.at<float>(j, 0) = (m2.at<float>(j, 0) - left) ;
			m2.at<float>(j, 1) = (m2.at<float>(j, 1) - top) ;
			// 第3列不变
		}
		cv::Mat kpts(Size(3, 17), CV_32FC1);
		cv::multiply(m2, m1, kpts);
		cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(1, 0), kpts.at<float>(1, 1)), color_tables[0], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(1, 0), kpts.at<float>(1, 1)), cv::Point(kpts.at<float>(3, 0), kpts.at<float>(3, 1)), color_tables[1], 2, 8, 0);

		// nose->right_eye->right_ear.(0, 2), (2, 4)
		cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(2, 0), kpts.at<float>(2, 1)), color_tables[2], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(2, 0), kpts.at<float>(2, 1)), cv::Point(kpts.at<float>(4, 0), kpts.at<float>(4, 1)), color_tables[3], 2, 8, 0);

		// nose->left_shoulder->left_elbow->left_wrist.(0, 5), (5, 7), (7, 9)
		cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(5, 0), kpts.at<float>(5, 1)), color_tables[4], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(5, 0), kpts.at<float>(5, 1)), cv::Point(kpts.at<float>(7, 0), kpts.at<float>(7, 1)), color_tables[5], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(7, 0), kpts.at<float>(7, 1)), cv::Point(kpts.at<float>(9, 0), kpts.at<float>(9, 1)), color_tables[6], 2, 8, 0);

		// nose->right_shoulder->right_elbow->right_wrist.(0, 6), (6, 8), (8, 10)
		cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(6, 0), kpts.at<float>(6, 1)), color_tables[7], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(6, 0), kpts.at<float>(6, 1)), cv::Point(kpts.at<float>(8, 0), kpts.at<float>(8, 1)), color_tables[8], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(8, 0), kpts.at<float>(8, 1)), cv::Point(kpts.at<float>(10, 0), kpts.at<float>(10, 1)), color_tables[9], 2, 8, 0);

		// left_shoulder->left_hip->left_knee->left_ankle.(5, 11), (11, 13), (13, 15)
		cv::line(frame, cv::Point(kpts.at<float>(5, 0), kpts.at<float>(5, 1)), cv::Point(kpts.at<float>(11, 0), kpts.at<float>(11, 1)), color_tables[10], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(11, 0), kpts.at<float>(11, 1)), cv::Point(kpts.at<float>(13, 0), kpts.at<float>(13, 1)), color_tables[11], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(13, 0), kpts.at<float>(13, 1)), cv::Point(kpts.at<float>(15, 0), kpts.at<float>(15, 1)), color_tables[12], 2, 8, 0);

		// right_shoulder->right_hip->right_knee->right_ankle.(6, 12), (12, 14), (14, 16)
		cv::line(frame, cv::Point(kpts.at<float>(6, 0), kpts.at<float>(6, 1)), cv::Point(kpts.at<float>(12, 0), kpts.at<float>(12, 1)), color_tables[13], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(12, 0), kpts.at<float>(12, 1)), cv::Point(kpts.at<float>(14, 0), kpts.at<float>(14, 1)), color_tables[14], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(14, 0), kpts.at<float>(14, 1)), cv::Point(kpts.at<float>(16, 0), kpts.at<float>(16, 1)), color_tables[16], 2, 8, 0);

		for (int row = 0; row < kpts.rows; row++) {
			int x = static_cast<int>(kpts.at<float>(row, 0));
			int y = static_cast<int>(kpts.at<float>(row, 1));
			cv::circle(frame, cv::Size(x, y), 3, color_tables[row], 4, 8, 0);
		}

	}
	// 计算并显示 FPS
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
YOLODetector::~YOLODetector() {
	this->session.release();
	this->env.release();
	this->session_options.release();
	std::cout << "destory instance done" << endl;
}
