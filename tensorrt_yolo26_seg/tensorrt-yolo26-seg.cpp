# include "yolo26-seg.h"
class Logger : public ILogger
{
	void log(Severity severity, const char* msg)  noexcept
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

YOLOv26TRTSegment::~YOLOv26TRTSegment() {
	// 同步结束，释放资源
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	// 1. 先释放CUDA缓冲区（最底层资源）
	for (int i = 0; i < 3; ++i) {
		if (buffers[i]) {
			cudaFree(buffers[i]);
			buffers[i] = nullptr;
		}
	}
	std::cout << "destory instance done!" << std::endl;
}

void YOLOv26TRTSegment::initConfig(std::string enginefile, std::string label_map) {
	std::ifstream fp(label_map);
	std::string name;
	std::cout << "CUDA devices: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
	while (getline(fp, name)) {
		if (!name.empty()) {
			this->classNames.push_back(name);
		}
	}
	fp.close();
	std::ifstream file(enginefile, std::ios::binary);
	std::vector<char> trtModelStream;
	int size = 0;
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream.resize(size);
		file.read(trtModelStream.data(), size);
		file.close();
	}

	// 初始化几个对象
	this->runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
	this->engine = std::unique_ptr<ICudaEngine>(this->runtime->deserializeCudaEngine(trtModelStream.data(), size));//将二进制数据转换为可执行的 CUDA 引擎
	this->context = std::unique_ptr<IExecutionContext>(this->engine->createExecutionContext());
	//输入信息
	auto input_shape = engine->getTensorShape("images");  // 通过张量名获取
	auto output_shape = engine->getTensorShape("output0");
	auto output_mask_shape = engine->getTensorShape("output1");
	this->input_h = input_shape.d[2];
	this->input_w = input_shape.d[3];
	printf("inputH : %d, inputW: %d \n", input_h, input_w);
	this->output_h = output_shape.d[1];
	this->output_w = output_shape.d[2];
	std::cout << "out data format: " << output_h << "x" << output_w << std::endl;
	//获得mask维度信息
	this->output_mask_weight = output_mask_shape.d[1];
	this->output_mask_h = output_mask_shape.d[2];
	this->output_mask_w = output_mask_shape.d[3];
	// 创建GPU显存输入/输出缓冲区
	cudaMalloc(&buffers[0], this->input_h * this->input_w * 3 * sizeof(float));
	cudaMalloc(&buffers[1], this->output_h * this->output_w * sizeof(float));
	cudaMalloc(&buffers[2], this->output_mask_weight * this->output_mask_h * this->output_mask_w * sizeof(float));
	// 创建输出结果
	this->prob = cv::Mat(Size(this->output_w, this->output_h), CV_32F);
	this->mprob = cv::Mat(Size(this->output_mask_h * this->output_mask_w, this->output_mask_weight), CV_32F);
	// 创建cuda流
	cudaStreamCreate(&stream);
}

void YOLOv26TRTSegment::detect(cv::Mat& frame) {
	cv::RNG rng(12345);
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
	float sx = 160.0f / _max;
	float sy = 160.0f / _max;
	float x_factor = gpu_image.cols / static_cast<float>(this->input_w);
	float y_factor = gpu_image.rows / static_cast<float>(this->input_h);
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
	// GPU显存到内存
	cudaMemcpyAsync(prob.ptr<float>(), buffers[1], this->output_h * this->output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(mprob.ptr<float>(), buffers[2], this->output_mask_weight * this->output_mask_h * this->output_mask_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

	// 后处理, 300x38
	cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());
	for (int i = 0; i < this->prob.rows; i++) {
		//detect
		float score = this->prob.at<float>(i, 4);
		// 置信度 0～1之间
		if (score > this->conf_thresholod)
		{
			float x1 = this->prob.at<float>(i, 0);
			float y1 = this->prob.at<float>(i, 1);
			float x2 = this->prob.at<float>(i, 2);
			float y2 = this->prob.at<float>(i, 3);
			int rx1 = static_cast<int>(x1 * x_factor);
			int ry1 = static_cast<int>(y1 * y_factor);
			int rx2 = static_cast<int>(x2 * x_factor);
			int ry2 = static_cast<int>(y2 * y_factor);
			int class_id = static_cast<int>(this->prob.at<float>(i, 5));
			cv::Rect box(rx1, ry1, rx2 - rx1, ry2 - ry1);
			rx1 = std::max(0, box.x);
			ry1 = std::max(0, box.y);
			rx2 = std::min(frame.cols, box.br().x);
			ry2 = std::min(frame.rows, box.br().y);
			cv::Mat m2 = this->prob.row(i).colRange(this->output_w - 32, this->output_w); // 32个掩码系数 [1x32]
			cv::Mat m = m2 * this->mprob;    //M2:1*32 mask1: 32*25600 M:1*25600 矩阵乘法[1x32] × [32x25600] = [1x25600]
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
			cv::putText(frame, this->classNames[class_id], cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}
	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::addWeighted(frame, 0.6, rgb_mask, 0.4, 0, frame);
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}