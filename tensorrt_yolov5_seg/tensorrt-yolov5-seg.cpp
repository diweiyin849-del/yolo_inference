#include "yolov5-seg.h"

class Logger : public ILogger
{
	void log(Severity severity, const char* msg)  noexcept
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

YOLOv5TRTSegment::~YOLOv5TRTSegment() {
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

void YOLOv5TRTSegment::initConfig(std::string enginefile) {
	std::ifstream file(enginefile, std::ios::binary);
	std::vector<char> trtModelStream;   // C++ 风格内存管理
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
	// 获取输入与输出维度
	auto input_shape = engine->getTensorShape("images");  // 通过张量名获取
	auto output_shape = engine->getTensorShape("output0");
	auto output_mask_shape = engine->getTensorShape("output1");
	this->input_h = input_shape.d[2];
	this->input_w = input_shape.d[3];
	printf("inputH : %d, inputW: %d \n", this->input_h, this->input_w);
	this->output_h = output_shape.d[1];
	this->output_w = output_shape.d[2];
	std::cout << "out data format: " << this->output_h << "x" << this->output_w << std::endl;
	//获得mask维度信息
	this->output_mask_weight = output_mask_shape.d[1];
	this->output_mask_h = output_mask_shape.d[2];
	this->output_mask_w = output_mask_shape.d[3];
	cudaMalloc(&buffers[0], this->input_h * this->input_w * 3 * sizeof(float));
	cudaMalloc(&buffers[1], this->output_h * this->output_w * sizeof(float));
	cudaMalloc(&buffers[2], this->output_mask_weight * this->output_mask_h * this->output_mask_w * sizeof(float));

	// 创建输出结果
	this->prob = cv::Mat(Size(output_w, output_h), CV_32F);   // Size(列, 行)
	this->mprob = cv::Mat(Size(160 * 160, 32), CV_32F);
	// 创建cuda流
	cudaStreamCreate(&stream);
}

float YOLOv5TRTSegment::sigmoid_function(float a)
{
	float b = 1. / (1. + exp(-a));
	return b;
}

void YOLOv5TRTSegment::detect(cv::Mat& frame, std::vector<DetectResult>& results) {
	cv::RNG rng;
	int64 start = cv::getTickCount();
	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	// HWC => CHW
	float x_factor = _max / static_cast<float>(this->input_w);
	float y_factor = _max / static_cast<float>(this->input_h);
	float sx = 160.0f / _max;
	float sy = 160.0f / _max;

	cv::Mat tensor = cv::dnn::blobFromImage(image, 1.0f / 255.f, cv::Size(this->input_w, this->input_h), cv::Scalar(), true);

	// 内存到GPU显存

	cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), this->input_h * this->input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

	// 推理
	context->executeV2(buffers);

	// GPU显存到内存
	cudaMemcpyAsync(this->prob.ptr(), buffers[1], this->output_h * this->output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(this->mprob.ptr(), buffers[2], this->output_mask_weight * this->output_mask_h * this->output_mask_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

	// 后处理, 1x25200x85
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<cv::Mat> masks;
	std::vector<float> confidences;
	cv::Mat det_output(this->output_h, this->output_w, CV_32F, (float*)prob.ptr());
	cv::Mat mask1(32, 25600, CV_32F, (float*)this->mprob.ptr());
	for (int i = 0; i < det_output.rows; i++) {
		float confidence = det_output.at<float>(i, 4);
		if (confidence < this->conf_thresholod) {
			continue;
		}
		cv::Mat classes_scores = det_output.row(i).colRange(5, this->output_w - 32);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

		// 置信度 0～1之间
		if (score > this->score_thresholod)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;
			cv::Mat mask2 = det_output.row(i).colRange(this->output_w - 32, this->output_w);
			boxes.push_back(box);
			masks.push_back(mask2);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());
	for (size_t i = 0; i < indexes.size(); i++) {
		DetectResult dr;
		int index = indexes[i];
		int idx = classIds[index];
		dr.box = boxes[index];
		dr.classId = idx;
		dr.conf = confidences[index];

		cv::Rect box = boxes[index];
		int x1 = std::max(0, box.x);
		int y1 = std::max(0, box.y);
		int x2 = std::min(frame.cols, box.br().x);
		int y2 = std::min(frame.rows, box.br().y);
		cv::Mat m2 = masks[index];
		cv::Mat m = m2 * mask1;
		for (int col = 0; col < m.cols; col++) {
			m.at<float>(0, col) = sigmoid_function(m.at<float>(0, col));
		}
		cv::Mat m1 = m.reshape(1, 160);
		int mx1 = std::max(0, int(x1 * sx));
		int mx2 = std::max(0, int(x2 * sx));
		int my1 = std::min(160, int(y1 * sy));
		int my2 = std::min(160, int(y2 * sy));

		cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
		cv::Mat rm, det_mask;

		cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));
		for (int r = 0; r < rm.rows; r++) {
			for (int c = 0; c < rm.cols; c++) {
				float pv = rm.at<float>(r, c);
				if (pv > 0.5) {
					rm.at<float>(r, c) = 1.0;
				}
				else {
					rm.at<float>(r, c) = 0.0;
				}
			}
		}
		rm = rm * rng.uniform(0, 255);
		rm.convertTo(det_mask, CV_8UC1);
		cv::Mat mask = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
		det_mask.copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
		add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);
		cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 1, 8);
		cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
			cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
		results.push_back(dr);
	}

	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::addWeighted(frame, 0.6, rgb_mask, 0.4, 0, frame);
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
