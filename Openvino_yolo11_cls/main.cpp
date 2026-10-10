#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudafilters.hpp>
#include<opencv2/cudaarithm.hpp>
#include<iostream>
#include<fstream>
using namespace std;
using namespace cv;
string label_map = "D:/yolo_models/imagenet_cls.txt";
int main() {
	vector<string>labels;
	std::ifstream fp(label_map);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			labels.push_back(name);
		}
	}
	fp.close();
	ov::Core core;
	std::string onnxpath = "D:/yolo_models/yolo11n-cls.onnx";
	auto model = core.compile_model(onnxpath, "CPU");

	ov::InferRequest request = model.create_infer_request();
	ov::Tensor input_tensor = request.get_input_tensor();

	// BGR2RGB, 0~1, HWC -> CHW, 
	cv::Mat frame = cv::imread("C:/Users/24150/Desktop/test2.jpg");

	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);
	// 设置输入
	size_t data_s = 3 * 224 * 224 * sizeof(float);
	memcpy(input_tensor.data<float>(), blob.ptr<float>(), data_s);

	//request.infer(); // 同步推理
	request.start_async(); // 异步推理
	request.wait();
	// 后处理
	auto output_tensor = request.get_output_tensor();
	cv::Mat prob(1, 1000, CV_32F, output_tensor.data<float>());
	double max_score;
	cv::Point max_Loc;
	cv::minMaxLoc(prob, 0, &max_score, 0, &max_Loc);
	std::cout << "class index : " << max_Loc.x << std::endl;
	putText(frame, labels[max_Loc.x], Point(25, 25), 2, 0.5, Scalar(255, 0, 0), 2);
	cv::imshow("yolo11n-cls图像分类 + OpenVINO2026", frame);
	cv::waitKey(0);
	cv::destroyAllWindows();
}