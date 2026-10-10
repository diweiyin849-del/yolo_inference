#include "yolo26-pose.h"
void YOLO26PoseDetector::initConfig(std::string onnxpath) {
	ov::Core ie;
	ov::CompiledModel compiled_model = ie.compile_model(onnxpath, "GPU");
	this->infer_request = compiled_model.create_infer_request();
	cv::RNG rng(12345);
	for (int i = 0; i < 17; i++) {
		int a = rng.uniform(0, 255);
		int b = rng.uniform(0, 255);
		int c = rng.uniform(0, 255);
		this->color_tables.push_back(cv::Scalar(a, b, c));
	}
}

void YOLO26PoseDetector::detect(cv::Mat& frame) {
	// 获取输入格式
	int64 start = cv::getTickCount();
	// 请求网络输入
	ov::Tensor input_tensor = infer_request.get_input_tensor();

	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	float x_factor = image.cols / static_cast<float>(input_w);
	float y_factor = image.rows / static_cast<float>(input_h);
	cv::Mat m1 = cv::Mat::zeros(cv::Size(3, 17), CV_32F);
	for (int i = 0; i < 17; i++) {
		m1.at<float>(i, 0) = x_factor;
		m1.at<float>(i, 1) = y_factor;
		m1.at<float>(i, 2) = 1.0f;
	}
	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false);
	memcpy(input_tensor.data<float>(), blob.ptr<float>(), data_s);

	// 推理与返回结果
	this->infer_request.infer();

	ov::Tensor output_tensor = this->infer_request.get_output_tensor();
	ov::Shape outputDims = output_tensor.get_shape();
	this->output_h = outputDims[1];
	this->output_w = outputDims[2];
	cv::Mat dout(this->output_h, this->output_w, CV_32F, (float*)output_tensor.data<float>());
	//300x57
	for (int i = 0; i < dout.rows; i++) {
		float score = dout.at<float>(i, 4);
		// 置信度 0～1之间
		if (score > this->score_t)
		{
			float x1 = dout.at<float>(i, 0);
			float y1 = dout.at<float>(i, 1);
			float x2 = dout.at<float>(i, 2);
			float y2 = dout.at<float>(i, 3);
			int rx1 = static_cast<int>(x1 * x_factor);
			int ry1 = static_cast<int>(y1 * y_factor);
			int rx2 = static_cast<int>(x2 * x_factor);
			int ry2 = static_cast<int>(y2 * y_factor);
			cv::Rect box(rx1, ry1, rx2 - rx1, ry2 - ry1);
			cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1, 8);
			cv::rectangle(frame, cv::Point(box.tl().x, box.tl().y - 20), cv::Point(box.br().x, box.tl().y), cv::Scalar(0, 255, 255), -1);
			cv::Mat one_kypts = dout.row(i).colRange(this->output_w - 51, this->output_w);
			cv::Mat m2 = one_kypts.reshape(1, 17);// 重塑（reshape）17x3 自动计算列数 
			cv::Mat kpts(cv::Size(3, 17), CV_32F);
			cv::multiply(m2, m1, kpts);
			cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(1, 0), kpts.at<float>(1, 1)), color_tables[0], 2, 8, 0);
			cv::line(frame, cv::Point(kpts.at<float>(1, 0), kpts.at<float>(1, 1)), cv::Point(kpts.at<float>(3, 0), kpts.at<float>(3, 1)), color_tables[1], 2, 8, 0);

			// nose->right_eye->right_ear.(0, 2), (2, 4)
			cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(2, 0), kpts.at<float>(2, 1)), color_tables[2], 2, 8, 0);
			cv::line(frame, cv::Point(kpts.at<float>(2, 0), kpts.at<float>(2, 1)), cv::Point(kpts.at<float>(4, 0), kpts.at<float>(4, 1)), color_tables[3], 2, 8, 0);

			// nose->left_shoulder->left_elbow->left_wrist.(0, 5), (5, 7), (7, 9)
			cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(5, 0), kpts.at<float>(5, 1)), color_tables[4], 2, 8, 0);
			cv::line(frame, cv::Point(kpts.at<float>(5, 0), kpts.at<float>(5, 1)), cv::Point(kpts.at<float>(7, 0), kpts.at<float>(7, 1)), color_tables[5], 2, 8, 0);
			cv::line(frame, cv::Point(kpts.at<float>(7, 0), kpts.at<float>(7, 1)), cv::Point(kpts.at<float>(9, 0), kpts.at<float>(9, 1)), color_tables[6], 2, 8, 0);

			// nose->right_shoulder->right_elbow->right_wrist.(0, 6), (6, 8), (8, 10)
			cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(6, 0), kpts.at<float>(6, 1)), color_tables[7], 2, 8, 0);
			cv::line(frame, cv::Point(kpts.at<float>(6, 0), kpts.at<float>(6, 1)), cv::Point(kpts.at<float>(8, 0), kpts.at<float>(8, 1)), color_tables[8], 2, 8, 0);
			cv::line(frame, cv::Point(kpts.at<float>(8, 0), kpts.at<float>(8, 1)), cv::Point(kpts.at<float>(10, 0), kpts.at<float>(10, 1)), color_tables[9], 2, 8, 0);

			// left_shoulder->left_hip->left_knee->left_ankle.(5, 11), (11, 13), (13, 15)
			cv::line(frame, cv::Point(kpts.at<float>(5, 0), kpts.at<float>(5, 1)), cv::Point(kpts.at<float>(11, 0), kpts.at<float>(11, 1)), color_tables[10], 2, 8, 0);
			cv::line(frame, cv::Point(kpts.at<float>(11, 0), kpts.at<float>(11, 1)), cv::Point(kpts.at<float>(13, 0), kpts.at<float>(13, 1)), color_tables[11], 2, 8, 0);
			cv::line(frame, cv::Point(kpts.at<float>(13, 0), kpts.at<float>(13, 1)), cv::Point(kpts.at<float>(15, 0), kpts.at<float>(15, 1)), color_tables[12], 2, 8, 0);

			// right_shoulder->right_hip->right_knee->right_ankle.(6, 12), (12, 14), (14, 16)
			cv::line(frame, cv::Point(kpts.at<float>(6, 0), kpts.at<float>(6, 1)), cv::Point(kpts.at<float>(12, 0), kpts.at<float>(12, 1)), color_tables[13], 2, 8, 0);
			cv::line(frame, cv::Point(kpts.at<float>(12, 0), kpts.at<float>(12, 1)), cv::Point(kpts.at<float>(14, 0), kpts.at<float>(14, 1)), color_tables[14], 2, 8, 0);
			cv::line(frame, cv::Point(kpts.at<float>(14, 0), kpts.at<float>(14, 1)), cv::Point(kpts.at<float>(16, 0), kpts.at<float>(16, 1)), color_tables[16], 2, 8, 0);
			for (int row = 0; row < kpts.rows; row++) {
				int x = static_cast<int>(kpts.at<float>(row, 0));
				int y = static_cast<int>(kpts.at<float>(row, 1));
				cv::circle(frame, cv::Size(x, y), 3, color_tables[row], 4, 8, 0);
			}
		}
	}
	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2, 8);
}
