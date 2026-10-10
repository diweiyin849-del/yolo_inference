#include"yolo11-obb.h"

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
	printf("inputH : %d, inputW: %d \n", input_h, input_w);
	this->output_h = output_shape.d[1];
	this->output_w = output_shape.d[2];
	std::cout << "out data format: " << output_h << "x" << output_w << std::endl;
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
	float xy_factor = image.rows / (float)this->input_h;
	cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0), true, false);
	// 内存到显存,GPU目标地址, CPU源地址,方向: CPU→GPU,异步流
	cudaMemcpyAsync(buffers[0], blob.ptr<float>(), 3 * this->input_h * this->input_w * sizeof(float), cudaMemcpyHostToDevice, stream);
	// GPU推理
	this->context->executeV2(buffers);
	// 显存到内存 blob.data：返回 uchar*，最底层的字节指针 blob.ptr<float>()：返回 float*，类型安全的float指针
	cudaMemcpyAsync(prob.ptr<float>(), buffers[1], this->output_h * this->output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
	// 后处理, 1x10383*20
	std::vector<Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<float> angles;
	cv::Mat det_output = this->prob.t(); // 转置变成10284*20
	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, this->output_h - 1);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 置信度 0～1之间
		if (score > this->score_threshold)
		{
			angles.push_back(det_output.at<float>(i, this->output_h - 1)); //最后一列是角度
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = ((cx - 0.5 * ow) * xy_factor);
			int y = ((cy - 0.5 * oh) * xy_factor);
			int width = (ow * xy_factor);
			int height = (oh * xy_factor);
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

	std::vector<cv::Point> point_pts;
	std::vector<cv::Point> rrt_pts;
	for (size_t i = 0; i < indexes.size(); i++) {
		point_pts.clear();
		rrt_pts.clear();
		int index = indexes[i];
		int idx = classIds[index];
		cv::Rect box = boxes[index];
		point_pts.push_back(box.tl());
		point_pts.push_back(cv::Point(box.x + box.width, box.y));
		point_pts.push_back(box.br());
		point_pts.push_back(cv::Point(box.x, box.y + box.height));
		getRotatedPts(rrt_pts, -angles[index], point_pts, box);
		for (int r = 0; r < 4; r++) {
			cv::line(frame, rrt_pts[r], rrt_pts[(r + 1) % 4], cv::Scalar(255, 0, 255), 2, 8);
		}
		int cx = box.x + box.width / 2;
		int cy = box.y + box.height / 2;
		cv::putText(frame, classNames[idx], cv::Point(cx, cy), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);
	}
	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2, 8);
}

//利用矩阵2*3变换得到obb的检测框;
void YOLOv81112TRTDetector::getRotatedPts(std::vector<cv::Point>& rrts, float angle, std::vector<cv::Point>& point_pts, cv::Rect& box) {
	cv::Mat M = cv::Mat::zeros(Size(3, 2), CV_32FC1);
	double alpha = cos(angle);   //cos   sin   (1 - cos) * center_x - sin * center_y
	double beta = sin(angle);   // -sin  cos   sin * center_x + (1 - cos) * center_y
	int center_x = box.x + box.width / 2;
	int center_y = box.y + box.height / 2;
	double tx = (1 - alpha) * center_x - beta * center_y;
	double ty = beta * center_x + (1 - alpha) * center_y;
	M.at<float>(0, 0) = alpha;
	M.at<float>(1, 1) = alpha;
	M.at<float>(0, 1) = beta;
	M.at<float>(1, 0) = -beta;
	M.at<float>(0, 2) = tx;
	M.at<float>(1, 2) = ty;
	for (cv::Point pt : point_pts) {
		float x0 = pt.x;
		float y0 = pt.y;
		float x = M.at<float>(0, 0) * x0 + M.at<float>(0, 1) * y0 + M.at<float>(0, 2); //本质矩阵线性变换;
		float y = M.at<float>(1, 0) * x0 + M.at<float>(1, 1) * y0 + M.at<float>(1, 2); //本质矩阵线性变换;
		rrts.push_back(cv::Point(x, y));
	}
}