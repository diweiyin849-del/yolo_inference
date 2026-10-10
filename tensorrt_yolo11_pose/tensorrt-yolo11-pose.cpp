# include "yolo11-pose.h"

class Logger : public ILogger
{
	void log(Severity severity, const char* msg)  noexcept
	{
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

YOLO11TRTPose::~YOLO11TRTPose() {
	// 同步结束，释放资源
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	// 1. 先释放CUDA缓冲区（最底层资源）
	for (int i = 0; i < 2; ++i) {
		if (buffers[i]) {
			cudaFree(buffers[i]);
			buffers[i] = nullptr;
		}
	}
	std::cout << "destory instance done!" << std::endl;
}

void YOLO11TRTPose::initConfig(std::string enginepath) {
	std::ifstream file(enginepath, std::ios::binary);
	vector<char>trtModelStream;
	int size = 0;
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream.resize(size);
		file.read(trtModelStream.data(), size);
		file.close();
	}
	this->runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
	this->engine = std::unique_ptr<ICudaEngine>(this->runtime->deserializeCudaEngine(trtModelStream.data(), size));//将二进制数据转换为可执行的 CUDA 引擎
	this->context = std::unique_ptr<IExecutionContext>(this->engine->createExecutionContext());

	// 获取输入维度信息
	auto input_shape = engine->getTensorShape("images");  // 通过张量名获取
	auto output_shape = engine->getTensorShape("output0");
	this->input_h = input_shape.d[2];
	this->input_w = input_shape.d[3];
	printf("inputH : %d, inputW: %d \n", input_h, input_w);
	this->output_h = output_shape.d[1];
	this->output_w = output_shape.d[2];
	std::cout << "out data format: " << output_h << "x" << output_w << std::endl;

	// 创建GPU显存输入/输出缓冲区
	cudaMalloc(&buffers[0], this->input_h * input_w * 3 * sizeof(float));
	cudaMalloc(&buffers[1], this->output_h * this->output_w * sizeof(float));

	// 创建临时缓存输出
	this->prob = cv::Mat(Size(this->output_w, this->output_h), CV_32FC1);;

	// 创建cuda流
	cudaStreamCreate(&stream);

	cv::RNG rng;
	for (int i = 0; i < 17; i++) {
		int a = rng.uniform(0, 255);
		int b = rng.uniform(0, 255);
		int c = rng.uniform(0, 255);
		this->color_tables.push_back(cv::Scalar(a, b, c));
	}
}

void YOLO11TRTPose::detect(cv::Mat& frame) {
	int64 start = cv::getTickCount();
	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	// 图象预处理 - 格式化操作
	cv::cuda::GpuMat gpu_frame;
	gpu_frame.upload(frame);
	cv::cuda::GpuMat gpu_image = cv::cuda::GpuMat(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	gpu_frame.copyTo(gpu_image(roi));
	float x_factor = gpu_image.cols / static_cast<float>(this->input_w);
	float y_factor = gpu_image.rows / static_cast<float>(this->input_h);
	//m1
	cv::Mat m1 = cv::Mat::zeros(cv::Size(3, 17), CV_32FC1);
	for (int i = 0; i < 17; i++) {
		m1.at<float>(i, 0) = x_factor;
		m1.at<float>(i, 1) = y_factor;
		m1.at<float>(i, 2) = 1.0f;
	}
	//Resize
	cv::cuda::GpuMat resized;
	cv::cuda::resize(gpu_image, resized, cv::Size(this->input_w, this->input_h), 0, 0, cv::INTER_LINEAR);
	//To float32 (0-1)
	cv::cuda::GpuMat float_image;
	resized.convertTo(float_image, CV_32FC3, 1.f / 255.f);
	// BGR To RGB
	cv::cuda::GpuMat rgb;
	cv::cuda::cvtColor(float_image, rgb, cv::COLOR_BGR2RGB);
	std::vector<cv::cuda::GpuMat> rgb_out;
	//cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
	//表示第 i 个 GpuMat 的起始地址指向 buffers[0] 偏移 i * input_w * input_h 个 float 元素的位置。
	//用户提供的 GPU 显存指针，GpuMat 不会自己分配显存，而是直接使用指定的 GPU 显存
	for (size_t i = 0; i < 3; ++i)
	{
		rgb_out.emplace_back(cv::cuda::GpuMat(cv::Size(this->input_w, this->input_h), CV_32FC1, (float*)buffers[0] + i * this->input_w * this->input_h));
	}
	// opencv HWC order -> CHW order
	cv::cuda::split(rgb, rgb_out);
	gpu_frame.release();
	resized.release();
	float_image.release();
	rgb.release();
	gpu_image.release();

	// 推理
	context->executeV2(buffers);

	// 显存到内存
	cudaMemcpyAsync(prob.ptr<float>(), buffers[1], this->output_h * this->output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

	// 后处理, 1x56x8400
	std::vector<cv::Rect> boxes;
	std::vector<cv::Mat> kypts;
	std::vector<float> confidences;
	cv::Mat det_output = this->prob.t(); // 8400x56
	for (int i = 0; i < det_output.rows; i++) {
		float conf = det_output.at<float>(i, 4);
		// 置信度 0～1之间
		if (conf > this->conf_t)
		{
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
			cv::Mat pts = det_output.row(i).colRange(5, this->output_h);
			kypts.push_back(pts);
			confidences.push_back(conf);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		DetectResult dr;
		int index = indexes[i];
		dr.score = confidences[index];
		cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 1, 8);
		cv::Mat one_kypts = kypts[index];
		cv::Mat m2 = one_kypts.reshape(1, 17);// 重塑（reshape）17x3 自动计算列数 
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
	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2, 8);
}