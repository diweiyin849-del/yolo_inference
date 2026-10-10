#include"yolo26-depth.h"

void YOLO11Detector::initConfig(std::string onnxpath) {

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
	float xy_factor = _max / (float)this->input_w;

	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(), true, false);
	//memcpy 将blob的数据拷贝到输入张量中。data_s 应该是预先计算好的数据大小（3 * input_h * input_w * sizeof(float)）。
	memcpy(input_tensor.data<float>(), blob.ptr<float>(), data_s);

	// 推理与返回结果
	this->infer_request.infer(); //同步执行推理，模型计算完成后返回。
	ov::Tensor output = this->infer_request.get_output_tensor();
	ov::Shape outputDims = output.get_shape();
	this->output_h = outputDims[2]; // 768
	this->output_w = outputDims[3]; // 768

	cv::Mat depth_map(this->output_h, this->output_w, CV_32F, (float*)output.data());
	
	// 2. (可选) 归一化深度图以便可视化
	cv::Mat depth_normalized;
	// 将深度值映射到 0-255 范围，用于显示
	cv::normalize(depth_map, depth_normalized, 0, 255, cv::NORM_MINMAX);//作用：将 depth_map 中的深度值线性缩放到 [0, 255] 区间。
	depth_normalized.convertTo(depth_normalized, CV_8UC1); //将 depth_normalized 的数据类型从 CV_32F（浮点）转换为 CV_8U

	// 3. 应用伪彩色图 (JET) 使其更直观
	cv::Mat depth_colored;
	cv::applyColorMap(depth_normalized, depth_colored, cv::COLORMAP_JET); //映射到BGR图像。蓝 → 青 → 黄 → 红。小值（近处）→ 蓝色 ,大值（远处）→ 红色

	// 4. 显示或保存结果
	cv::imshow("Depth Map", depth_colored);

	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
