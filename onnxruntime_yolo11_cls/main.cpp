#include "onnxruntime_cxx_api.h"
#include"cpu_provider_factory.h"
#include <iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudafilters.hpp>
#include<opencv2/cudaarithm.hpp>
#include<fstream>
using namespace std;
using namespace cv;
string label_map = "D:/yolo_models/imagenet_cls.txt";
int main() {
	cv::Mat frame = cv::imread("C:/Users/24150/Desktop/test2.jpg");
	vector<string>labels;
	std::ifstream fp(label_map);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			labels.push_back(name);
		}
	}
	fp.close();

	std::string onnxpath = "D:/yolo_models/yolo11n-cls.onnx";
	std::wstring modelpath = std::wstring(onnxpath.begin(), onnxpath.end());
	Ort::SessionOptions session_options;  //用于配置推理会话（Session），比如优化等级、使用的执行提供者（CPU/CUDA）等。
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolo11n-cls-ONNX");//创建 ONNX Runtime环境,第一个参数是日志级别,第二个参数是自定义的日志标识名称
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);//设置图优化等级,ORT_ENABLE_BASIC 表示启用基础优化,通常能提升推理速度。
	std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
	OrtStatusPtr status = OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0); //为 session_options 添加 CUDA 执行提供者（即使用 GPU 推理）
	Ort::Session session_(env, modelpath.c_str(), session_options);//创建推理会话（Session）。传入环境、模型路径、以及配置好的 session_options。
	int input_nodes_num = session_.GetInputCount();
	int output_nodes_num = session_.GetOutputCount();
	std::vector<std::string>input_node_names; //用于存储输入节点的名称
	std::vector<std::string>output_node_names;//用于存储输出节点的名称
	Ort::AllocatorWithDefaultOptions allocator;
	int input_w = 0; int output_w = 0;
	int input_h = 0; int output_h = 0;
	for (int i = 0; i < input_nodes_num; i++) {  //GetInputNameAllocated(i, allocator)：获取第 i 个输入节点的名称，返回一个 Ort::AllocatedStringPtr管理字符串
		Ort::AllocatedStringPtr input_name = session_.GetInputNameAllocated(i, allocator);  //Ort::AllocatedStringPtr（可看作字符串的智能指针）。
		input_node_names.push_back(input_name.get());   //.get()：获取分配器内部的 char* 字符串。
		std::vector<int64_t> inputShapeInfo = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		int ch = inputShapeInfo[1];
		input_h = inputShapeInfo[2];
		input_w = inputShapeInfo[3];
		std::cout << "input format:" << ch << "x" << input_h << "x" << input_w << std::endl;
	}
	for (int i = 0; i < output_nodes_num; i++) {
		Ort::AllocatedStringPtr output_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());
		std::vector<int64_t> outputShapeInfo = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		output_h = outputShapeInfo[0];
		output_w = outputShapeInfo[1];
		std::cout << "output format:" << output_h << "x" << output_w << std::endl;
	}

	//左上角填充预处理;
	/*
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	cv::Mat tensor = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);
	*/

	//YOLOlettlebox预处理；
	float scale = std::min(static_cast<float>(input_w) / frame.cols, static_cast<float>(input_h) / frame.rows);
	int new_w = static_cast<int>(frame.cols * scale);
	int new_h = static_cast<int>(frame.rows * scale);
	cv::Mat resized;
	cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
	cv::Mat canvas = cv::Mat::zeros(cv::Size(input_w, input_h), CV_8UC3);
	int left = (input_w - new_w) / 2;
	int top = (input_h - new_h) / 2;
	resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));
	cv::Mat tensor = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);
	// 创建 ONNX Runtime 张量Ort::Value
	size_t tpixels = 3 * input_h * input_w;
	std::array<int64_t, 4>input_shape_info{ 1,3,input_h,input_w };
	Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); //创建 CPU 内存信息对象，用于告诉 ONNX Runtime 张量数据存储在 CPU 的哪个区域。
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, tensor.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
	const std::array<const char*, 1>inputNames = { input_node_names[0].c_str() }; //准备输入名称数组
	const std::array<const char*, 1>outputNames = { output_node_names[0].c_str() };//准备输出名称数组

	//推理
	std::vector<Ort::Value>ort_outputs;
	try {
		ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, inputNames.size(), outputNames.data(), outputNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}
	float* pdata = ort_outputs[0].GetTensorMutableData<float>(); //成员函数 GetTensorMutableData<float>() 返回了一个 float*指向输出的value张量数据;
	//后处理;
	cv::Mat prob(output_h, output_w, CV_32F, (float*)pdata);
	double max_score;
	cv::Point max_Loc;
	cv::minMaxLoc(prob, 0, &max_score, 0, &max_Loc);
	std::cout << "class index : " << max_Loc.x << std::endl;
	putText(frame, labels[max_Loc.x], Point(25, 25), 2, 0.5, Scalar(255, 0, 0), 2);
	cv::imshow("YOLO11n-cls + ONNX 1.22", frame);
	session_options.release();
	session_.release();
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

