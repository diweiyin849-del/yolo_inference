#include "openvino_queue_async_detect.h"

void YOLO11CallbackAsyncDetector::initConfig(std::string onnxpath, std::string labelpath) {
	std::ifstream fp(labelpath);
	std::string name;
	while (getline(fp, name)) {
		if (!name.empty()) {
			this->classNames.push_back(name);
		}
	}

	ov::Core core;
	ov::CompiledModel compiled_model = core.compile_model(onnxpath, "GPU");
	this->request1 = compiled_model.create_infer_request();
	this->request2 = compiled_model.create_infer_request();
	this->request3 = compiled_model.create_infer_request();
	this->request4 = compiled_model.create_infer_request();
	request_queue.emplace_back(this->request1);
	request_queue.emplace_back(this->request2);
	request_queue.emplace_back(this->request3);
	request_queue.emplace_back(this->request4);

	//infer_request绑定后处理;
	std::exception_ptr exception_var;  //声明一个异常指针变量，用于捕获异步回调中发生的异常。
	this->request1.set_callback([&](std::exception_ptr ex) {
		if (ex) {
			exception_var = ex;
			std::cerr << "Inference failed!" << std::endl;
			return;
		}
		ov::Tensor output = this->request1.get_output_tensor();
		this->postProcess(output);
		});
	this->request2.set_callback([&](std::exception_ptr ex) {
		if (ex) {
			exception_var = ex;
			std::cerr << "Inference failed!" << std::endl;
			return;
		}
		ov::Tensor output = this->request2.get_output_tensor();
		this->postProcess(output);
		});
	this->request3.set_callback([&](std::exception_ptr ex) {
		if (ex) {
			exception_var = ex;
			std::cerr << "Inference failed!" << std::endl;
			return;
		}
		ov::Tensor output = this->request3.get_output_tensor();
		this->postProcess(output);
		});
	this->request4.set_callback([&](std::exception_ptr ex) {
		if (ex) {
			exception_var = ex;
			std::cerr << "Inference failed!" << std::endl;
			return;
		}
		ov::Tensor output = this->request4.get_output_tensor();
		this->postProcess(output);
		});
	std::cout << "end the init config..." << std::endl;
}


void YOLO11CallbackAsyncDetector::propreoces_frame(ov::Tensor& input_tensor, cv::Mat& img) {
	ov::Shape tensor_shape = input_tensor.get_shape();
	size_t input_h = tensor_shape[2];
	size_t input_w = tensor_shape[3];
	int w = img.cols;
	int h = img.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	img.copyTo(image(roi));
	this->xy_factor = image.cols / static_cast<float>(input_w);
	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false);
	memcpy(input_tensor.data<float>(), blob.ptr<float>(), this->data_s);
}

void YOLO11CallbackAsyncDetector::detect(std::string source) {
	cv::VideoCapture capture(source);
	cv::Mat frame, next_frame;
	capture.read(frame);
	auto input_tensor = this->request_queue[0].get_input_tensor();
	propreoces_frame(input_tensor, frame);
	this->request_queue[0].start_async();
	int sindex = 0;
	while (true) {
		bool ret = capture.read(next_frame);
		if (next_frame.empty()) {
			break;
		}
		int64 start = cv::getTickCount();
		int rid = (sindex % this->numofjobs);
		if (rid == 0) {
			auto input_tensor = this->request_queue[rid + 1].get_input_tensor();
			propreoces_frame(input_tensor, next_frame);
			this->request_queue[rid + 1].start_async();
			this->request_queue[rid].wait();    //wait() 可以保证推理和后处理都已经执行完毕。
			for (size_t t = 0; t < det_boxes.size(); t++) {
				int idx = det_ids[t];
				cv::rectangle(frame, det_boxes[t], Scalar(0, 0, 255), 2, 8, 0);
				putText(frame, classNames[idx].c_str(), det_boxes[t].tl(), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 1, 8);
			}
		}
		else if (rid == 1) {
			auto input_tensor = this->request_queue[rid + 1].get_input_tensor();
			propreoces_frame(input_tensor, next_frame);
			this->request_queue[rid + 1].start_async();
			this->request_queue[rid].wait();
			for (size_t t = 0; t < det_boxes.size(); t++) {
				int idx = det_ids[t];
				cv::rectangle(frame, det_boxes[t], Scalar(0, 0, 255), 2, 8, 0);
				putText(frame, classNames[idx].c_str(), det_boxes[t].tl(), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 1, 8);
			}
		}
		else if (rid == 2) {
			auto input_tensor = this->request_queue[rid + 1].get_input_tensor();
			propreoces_frame(input_tensor, next_frame);
			this->request_queue[rid + 1].start_async();
			this->request_queue[rid].wait();
			for (size_t t = 0; t < det_boxes.size(); t++) {
				int idx = det_ids[t];
				cv::rectangle(frame, det_boxes[t], Scalar(0, 0, 255), 2, 8, 0);
				putText(frame, classNames[idx].c_str(), det_boxes[t].tl(), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 1, 8);
			}
		}
		else if (rid == 3) {
			auto input_tensor = this->request_queue[0].get_input_tensor();
			propreoces_frame(input_tensor, next_frame);
			this->request_queue[0].start_async();
			this->request_queue[rid].wait();
			for (size_t t = 0; t < det_boxes.size(); t++) {
				int idx = det_ids[t];
				cv::rectangle(frame, det_boxes[t], Scalar(0, 0, 255), 2, 8, 0);
				putText(frame, classNames[idx].c_str(), det_boxes[t].tl(), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 1, 8);
			}
		}

		//     FPS render it
		float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
		putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
		cv::imshow("YOLO11+OpenVINO2026异步回调流水线", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
		next_frame.copyTo(frame);
		sindex++;
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return;
}

void YOLO11CallbackAsyncDetector::postProcess(ov::Tensor& output) {
	det_ids.clear();
	det_boxes.clear();
	const ov::Shape outputDims = output.get_shape();
	size_t numRows = outputDims[1];
	size_t numCols = outputDims[2];
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	cv::Mat dout(numRows, numCols, CV_32F, (float*)output.data<float>());
	cv::Mat det_output = dout.t();
	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, det_output.cols);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		if (score > this->confidence_t)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * xy_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * xy_factor);
			int width = static_cast<int>(ow * xy_factor);
			int height = static_cast<int>(oh * xy_factor);
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
		int index = indexes[i];
		det_ids.emplace_back(classIds[index]);
		det_boxes.emplace_back(boxes[index]);
	}
}

