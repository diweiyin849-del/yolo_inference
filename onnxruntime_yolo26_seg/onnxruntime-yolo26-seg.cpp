#include "yolo26-seg.h"

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
	Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator); //这里的0/1是index
	this->output_node_names.push_back(output_name_ptr.get());
	Ort::AllocatedStringPtr output_mask_ptr = session.GetOutputNameAllocated(1, allocator);
	this->output_node_names.push_back(output_mask_ptr.get());

	this->inputNames = { this->input_node_names[0].c_str() }; //准备输入名称数组
	this->outputNames = { this->output_node_names[0].c_str(),this->output_node_names[1].c_str() };//准备输出名称数组
	// 可选：打印输入输出形状
	std::vector<int64_t> input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "Input shape: " << input_shape[1] << "x" << input_shape[2] << "x" << input_shape[3] << std::endl;
	std::vector<int64_t> output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "Output shape: " << output_shape[1] << "x" << output_shape[2] << std::endl;
	std::vector<int64_t> mask_shape = session.GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "maskput shape: " << mask_shape[1] << "x" << mask_shape[2] << "x" << mask_shape[3] << std::endl;
	this->output_h = output_shape[1];
	this->output_w = output_shape[2];
	this->output_mask_weight = mask_shape[1];
	this->output_mask_h = mask_shape[2];
	this->output_mask_w = mask_shape[3];
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
	float xy_factor = static_cast<float>(_max) / static_cast<float>(input_w);
	float sx = 160.0f / _max;
	float sy = 160.0f / _max;
	// blobFromImage: 缩放 + 归一化(1/255) + 转为NCHW + BGR->RGB
	cv::Mat tensor = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(), true, false);

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
	float* raw_output = outputs[0].GetTensorMutableData<float>();
	float* mask_output = outputs[1].GetTensorMutableData<float>();

	cv::Mat mask1(this->output_mask_weight, this->output_mask_h * this->output_mask_w, CV_32F, (float*)mask_output);
	cv::Mat dout(this->output_h, this->output_w, CV_32F, (float*)raw_output);
	// 后处理, 300x38
	cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());
	for (int i = 0; i < dout.rows; i++) {
		//detect
		float score = dout.at<float>(i, 4);
		// 置信度 0～1之间
		if (score > this->conf_t)
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
	cv::putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}

YOLODetector::~YOLODetector() {
	this->session.release();
	this->env.release();
	this->session_options.release();
	std::cout << "destory instance done" << endl;
}