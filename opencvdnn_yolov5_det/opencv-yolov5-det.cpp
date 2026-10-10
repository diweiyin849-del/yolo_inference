#include "yolov5-det.h"

void YOLODetector::initConfig(std::string onnxpath, std::string textpath) {
    // 1. 读取类别文件
    std::ifstream fp(textpath);
    std::string name;
    while (std::getline(fp, name)) {
        if (!name.empty()) {
            labels.push_back(name);
        }
    }
    fp.close();
    this->net = cv::dnn::readNetFromONNX(onnxpath);
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    //this->net.enableFusion(false);
    //this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

void YOLODetector::detect(cv::Mat& frame) {
    int64 start = cv::getTickCount();
    // -------------------- 预处理 --------------------
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    // letterbox: 创建正方形画布，填充黑色
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));
    float xy_factor = static_cast<float>(_max) / static_cast<float>(this->input_w);
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(), true, false);
    this->net.setInput(blob);
    cv::Mat preds = this->net.forward();
    this->output_h = preds.size[1];
    this->output_w = preds.size[2];
    cv::Mat dout(this->output_h, this->output_w, CV_32F, (float*)preds.ptr());

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    for (int i = 0; i < dout.rows; ++i) {
        float obj_conf = dout.at<float>(i, 4);
        if (obj_conf < conf_t) continue;
        // 类别分数从第5列开始
        cv::Mat scores = dout.row(i).colRange(5, this->output_w);
        double max_score;
        cv::Point max_loc;
        cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &max_loc);
        int class_id = max_loc.x;
        if (max_score > 0.25) {
            float cx = dout.at<float>(i, 0);
            float cy = dout.at<float>(i, 1);
            float ow = dout.at<float>(i, 2);
            float oh = dout.at<float>(i, 3);
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
            classIds.push_back(class_id);
            confidences.push_back(max_score);
        }
    }

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, conf_t, 0.45, indexes);
    for (int idx : indexes) {
        int cid = classIds[idx];
        cv::rectangle(frame, boxes[idx], cv::Scalar(0, 0, 255), 2, 8);
        cv::putText(frame, labels[cid], boxes[idx].tl(), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 1, 8);
    }
    // 计算并显示 FPS
    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    cv::putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
