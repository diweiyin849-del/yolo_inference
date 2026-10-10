#include "openvino-async-detect.h"

void YOLO11AsyncDetector::initConfig(std::string onnxpath, std::string textpath) {
	std::ifstream fp(textpath);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			labels.push_back(name);
		}
	}
	fp.close();
	ov::Core ie;
	ov::CompiledModel compiled_model = ie.compile_model(onnxpath, "GPU");
	ov::InferRequest request1 = compiled_model.create_infer_request();
	ov::InferRequest request2 = compiled_model.create_infer_request();
	inferreqs.push_back(request1);
	inferreqs.push_back(request2);
}

void YOLO11AsyncDetector::preprocess(cv::Mat& frame, ov::Tensor& input_tensor, float& xy_factor) {
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	xy_factor = _max / (float)input_w;
	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false);
	memcpy(input_tensor.data<float>(), blob.ptr<float>(), data_s);
}


void YOLO11AsyncDetector::detect(std::string videofile) {
	cv::VideoCapture capture(videofile);
	cv::Mat frame, nextFrame;
	float xy_float;
	capture.read(frame);
	ov::Tensor input_tensor = inferreqs[0].get_input_tensor();
	preprocess(frame, input_tensor, xy_float);
	inferreqs[0].start_async();

	while (true) {
		capture.read(nextFrame);
		if (nextFrame.empty()) {
			break;
		}
		int64 start = cv::getTickCount();
		ov::Tensor input_tensor = inferreqs[1].get_input_tensor();
		// 图象预处理 - 格式化操作
		preprocess(nextFrame, input_tensor, xy_float); //下一帧的预处理可以和当前帧的推理 重叠执行
		// inferreqs[0] 总是那个“已启动上一帧推理、需要等待其完成”的对象，inferreqs[1] 总是那个“用来启动当前帧推理”的对象。
		// 推理与返回结果
		inferreqs[1].start_async();
		inferreqs[0].wait();
		auto output = inferreqs[0].get_output_tensor();
		auto outputDims = output.get_shape();
		this->output_h = outputDims[1]; // 84
		this->output_w = outputDims[2]; //8400
		cv::Mat dout(this->output_h, this->output_w, CV_32F, (float*)output.data<float>());
		cv::Mat det_output = dout.t();
		postProcess(det_output, frame, xy_float);

		// 计算FPS render it
		float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
		putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
		cv::imshow("YOLO11+OpenVINO2026异步流水线", frame);
		if (cv::waitKey(1) == 27)
		{
			break;
		}
		//为了统一循环内的代码逻辑（总是用固定的索引1做启动，索引0做等待），我们可以交换两个对象的角色，
		// 让它们重新回到上一轮开始时的状态：inferreqs[0] 总是那个“已启动推理、需要等待其完成”的对象，inferreqs[1] 总是那个“用来启动下一帧推理”的对象。
		auto temp = inferreqs[0];
		inferreqs[0] = inferreqs[1];
		inferreqs[1] = temp;
		nextFrame.copyTo(frame);
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
}


void YOLO11AsyncDetector::postProcess(cv::Mat& det_output, cv::Mat& frame, float xy_factor) {
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, this->output_h);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 置信度 0～1之间
		if (score > this->score_t)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = ((cx - 0.5 * ow) * xy_factor);
			int y = ((cy - 0.5 * oh) * xy_factor);
			int width = (ow * xy_factor);
			int height = (oh * xy_factor);
			cv::Rect box(x, y, width, height);
			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}
	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];
		cv::Rect box = boxes[index];
		cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2, 8);
		cv::putText(frame, labels[idx], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0), 1);
	}
}
