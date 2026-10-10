# include "yolov5-det.h"
class Logger : public ILogger
{
	void log(Severity severity, const char* msg)  noexcept
	{
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

YOLOv5TRTDetector::~YOLOv5TRTDetector() {
	//同步结束,释放资源
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

void YOLOv5TRTDetector::initConfig(std::string enginefile) {
	std::ifstream file(enginefile, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open engine file: " << enginefile << std::endl;
		return;
	}
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

	// 初始化几个对象
	//Iruntime从文件读取的二进制数据（trtModelStream）反序列化为 CUDA 引擎
	//Iengine从引擎创建一个 执行上下文（IExecutionContext）。
	// IExecutionContext执行上下文负责具体的推理任务
	this->runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
	this->engine = std::unique_ptr<ICudaEngine>(this->runtime->deserializeCudaEngine(trtModelStream.data(), size));//将二进制数据转换为可执行的 CUDA 引擎
	this->context = std::unique_ptr<IExecutionContext>(this->engine->createExecutionContext());

	// 获取输入维度信息
	auto input_shape = engine->getTensorShape("images");  // 通过张量名获取
	auto output_shape = engine->getTensorShape("output0");
	this->input_h = input_shape.d[2];
	this->input_w = input_shape.d[3];
	printf("inputH : %d, inputW: %d \n", this->input_h, this->input_w);
	this->output_h = output_shape.d[1];
	this->output_w = output_shape.d[2];
	std::cout << "out data format: " << this->output_h << "x" << this->output_w << std::endl;
	// 创建GPU显存输入/输出缓冲区
	cudaMalloc(&buffers[0], this->input_h * this->input_w * 3 * sizeof(float));
	cudaMalloc(&buffers[1], this->output_h * this->output_w * sizeof(float));
	// 创建临时缓存输出
	this->prob = cv::Mat(Size(this->output_w, this->output_h), CV_32F);
	// 创建cuda流
	cudaStreamCreate(&stream);
}

void YOLOv5TRTDetector::detect(cv::Mat& frame, std::vector<DetectResult>& results) {
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
	// 后处理, 1x25200x85
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	int k = 0;
	for (int i = 0; i < prob.rows; i++) {
		float confidence = prob.at<float>(i, 4);
		if (confidence < this->conf_thresholod) {
			continue;
		}
		cv::Mat classes_scores = prob.row(i).colRange(5, this->output_w);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 置信度 0～1之间
		if (score > this->score_thresholod)
		{
			float cx = prob.at<float>(i, 0);
			float cy = prob.at<float>(i, 1);
			float ow = prob.at<float>(i, 2);
			float oh = prob.at<float>(i, 3);
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
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		int idx = indexes[i];
		DetectResult dr;
		dr.classId = classIds[idx];
		dr.conf = confidences[idx];
		dr.box = boxes[idx];
		cv::rectangle(frame, boxes[idx], cv::Scalar(0, 0, 255), 3, 8);
		cv::rectangle(frame, cv::Point(boxes[idx].tl().x, boxes[idx].tl().y - 20), cv::Point(boxes[idx].br().x, boxes[idx].tl().y), cv::Scalar(0, 255, 255), -1);
		results.push_back(dr);
	}
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}