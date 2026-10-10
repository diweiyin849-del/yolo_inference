#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <NvOnnxParser.h>
using namespace nvinfer1;
using namespace cv;
using namespace std;
class Logger : public ILogger
{
	void log(Severity severity, const char* msg)  noexcept
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

std::string labels_txt_file = "D:/yolo_models/imagenet_cls.txt";
std::vector<std::string> readClassNames();
std::vector<std::string> readClassNames()
{
	std::vector<std::string> classNames;
	std::ifstream fp(labels_txt_file);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			classNames.push_back(name);
		}
	}
	fp.close();
	return classNames;
}

int main(int argc, char** argv) {
	std::vector<std::string> labels = readClassNames();
	std::string enginepath = "D:/TensorRT-10.8.0.43/bin/yolo26n-cls-fp16.engine";
	std::ifstream file(enginepath, std::ios::binary);
	std::vector<char>trtModelStream;
	int size = 0;
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream.resize(size);
		file.read(trtModelStream.data(), size);
		file.close();
	}

	auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
	auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream.data(), size));//将二进制数据转换为可执行的 CUDA 引擎
	auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
	void* buffers[2] = { NULL, NULL };
	cudaStream_t stream;

	// 获取输入维度信息
	auto input_shape = engine->getTensorShape("images");  // 通过张量名获取
	auto output_shape = engine->getTensorShape("output0");
	int input_h = input_shape.d[2];
	int input_w = input_shape.d[3];
	printf("inputH : %d, inputW: %d \n", input_h, input_w);
	int output_h = output_shape.d[0];
	int output_w = output_shape.d[1];
	std::cout << "out data format: " << output_h << "x" << output_w << std::endl;

	cudaMalloc(&buffers[0], input_h * input_w * 3 * sizeof(float));
	cudaMalloc(&buffers[1], output_h * output_w * sizeof(float));
	Mat prob = cv::Mat(Size(output_w, output_h), CV_32FC1);
	//cuda  
	cudaStreamCreate(&stream);

	cv::Mat frame = cv::imread("C:/Users/24150/Desktop/test2.jpg");
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	cv::Mat tensor = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);

	cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

	context->executeV2(buffers);

	cudaMemcpyAsync(prob.ptr(), buffers[1], output_h * output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

	cv::Point maxL, minL;
	double maxv, minv;
	cv::minMaxLoc(prob, &minv, &maxv, &minL, &maxL);
	int max_index = maxL.x;
	std::cout << "label id: " << max_index << std::endl;
	cv::putText(frame, labels[max_index], cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);
	cv::imshow("YOLO26n-cls+tensorrt10.8", frame);
	cv::waitKey(0);
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
	return 0;
}