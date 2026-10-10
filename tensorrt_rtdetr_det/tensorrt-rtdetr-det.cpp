#include"rtdetr-det.h"

class Logger : public ILogger
{
	void log(Severity severity, const char* msg) noexcept
	{
		// suppress info-level messages msg 是一个C风格字符串指针（const char*），它包含了具体的日志消息内容。
		//继承自TensorRT的ILogger接口 重写log函数，过滤掉kINFO级别的日志，只输出警告和错误 创建了全局实例gLogger供后续使用
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

YOLOv81112TRTDetector::~YOLOv81112TRTDetector() {
	// 同步结束，释放资源
	cudaStreamSynchronize(stream); // 在销毁流之前，必须确保流中所有已提交的操作都已经完成，否则可能导致未定义行为（例如在操作尚未完成时释放了相关的资源）
	cudaStreamDestroy(stream); // 销毁CUDA流 先同步，再释放
	if (buffers[0]) {
		cudaFree(buffers[0]);  // 释放GPU输入缓冲区
		buffers[0] = nullptr;
	}
	if (buffers[1]) {
		cudaFree(buffers[1]);  // 释放GPU输出缓冲区
		buffers[1] = nullptr;
	}
	std::cout << "destory instance done!" << std::endl;
}

void YOLOv81112TRTDetector::initConfig(std::string enginepath, std::string label_map) {
	std::ifstream fp(label_map);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			this->classNames.push_back(name);
		}
	}
	fp.close();

	std::ifstream file(enginepath, std::ios::binary);//std::ios::binary: 以二进制模式打开.engine文件
	std::vector<char> trtModelStream;   // C++ 风格内存管理
	int size = 0;
	if (file.good()) {
		file.seekg(0, file.end);// 移动到文件末尾 + 0字节
		size = file.tellg();// 返回当前文件指针的位置（字节数）
		file.seekg(0, file.beg);
		trtModelStream.resize(size);
		file.read(trtModelStream.data(), size);  //从文件读取指定数量的字节到缓冲区, 参数1:目标缓冲区指针, 参数2：要读取的字节数   .data()返回从首地址开始的指针;
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
	printf("inputH : %d, inputW: %d \n", this->input_h, this->input_w);
	this->output_h = output_shape.d[1];
	this->output_w = output_shape.d[2];
	std::cout << "out data format: " << this->output_h << "x" << this->output_w << std::endl;
	// 创建GPU显存输入/输出缓冲区
	cudaMalloc(&buffers[0], this->input_h * this->input_w * 3 * sizeof(float));//输入也是float 1*3*640*640
	cudaMalloc(&buffers[1], this->output_h * this->output_w * sizeof(float));// TensorRT保证输出是float类型数据，不是图像没有通道不*3 1*84*8400
	// 创建临时缓存输出
	this->prob = cv::Mat(Size(this->output_w, this->output_h), CV_32F);
	// 创建cuda流
	cudaStreamCreate(&stream);
}

void YOLOv81112TRTDetector::detect(cv::Mat& frame) {
	int64 start = cv::getTickCount();
	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	float x_factor = image.cols / (float)this->input_w;
	float y_factor = image.rows / (float)this->input_h;
	cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0), true, false);
	// 内存到显存,GPU目标地址, CPU源地址,方向: CPU→GPU,异步流
	cudaMemcpyAsync(buffers[0], blob.ptr<float>(), 3 * this->input_h * this->input_w * sizeof(float), cudaMemcpyHostToDevice, stream);
	// GPU推理
	this->context->executeV2(buffers);
	// 显存到内存 blob.data：返回 uchar*，最底层的字节指针 blob.ptr<float>()：返回 float*，类型安全的float指针
	cudaMemcpyAsync(prob.ptr<float>(), buffers[1], this->output_h * this->output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
	// 后处理, 1x300x6
	cv::Mat det_output(this->output_h, this->output_w, CV_32F, prob.ptr<float>());  //格式化
	for (int i = 0; i < det_output.rows; i++) {
		double score = det_output.at<float>(i, 4);
		// 置信度 0～1之间
		if (score > this->score_threshold)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = ((cx - 0.5 * ow) * _max);
			int y = ((cy - 0.5 * oh) * _max);
			int width = (ow * _max);
			int height = (oh * _max);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;
			int classnum = det_output.at<float>(i, 5);
			cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1, 8);
			cv::rectangle(frame, cv::Point(box.tl().x, box.tl().y - 20), cv::Point(box.br().x, box.tl().y), cv::Scalar(0, 255, 255), -1);
			cv::putText(frame, this->classNames[classnum], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}
	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2, 8);
}