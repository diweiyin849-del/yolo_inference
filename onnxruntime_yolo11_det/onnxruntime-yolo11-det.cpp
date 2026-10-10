#include "yolo11-det.h"

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

    this->session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // 3. 使用 GPU（可切换 CPU）
    std::cout << "Using GPU device 0" << std::endl;
    //OrtStatusPtr status1 = OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);
    //OrtStatusPtr status1 = OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0);
    OrtStatusPtr status = OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0); //为 session_options 添加 CUDA 执行提供者（即使用 GPU 推理）
    std::wstring wpath(onnxpath.begin(), onnxpath.end());
    this->session = Ort::Session(this->env, wpath.c_str(), session_options);

    // 5. 获取输入输出节点名称（假设各只有一个）
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator); //输入名称和对应索引
    this->input_node_names.push_back(input_name_ptr.get());
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);//输出名称和对应索引
    this->output_node_names.push_back(output_name_ptr.get());

    this->inputNames = { input_node_names[0].c_str() }; //准备输入名称数组,c风格
    this->outputNames = { output_node_names[0].c_str() };//准备输出名称数组,c风格
    // 可选：打印输入输出形状
    std::vector<int64_t> input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Input shape: " << input_shape[1] << "x" << input_shape[2] << "x" << input_shape[3] << std::endl;
    std::vector<int64_t> output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output shape: " << output_shape[1] << "x" << output_shape[2] << std::endl;
    this->output_h = output_shape[1];
    this->output_w = output_shape[2];
}

void YOLODetector::detect(cv::Mat& frame) {
    int64 start = cv::getTickCount();
    // -------------------- 左上角填充预处理 --------------------
    /*
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    // letterbox: 创建正方形画布，填充黑色
    cv::Mat canvas = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(canvas(roi));
    float xy_factor = static_cast<float>(_max) / static_cast<float>(this->input_w);
    // blobFromImage: 缩放 + 归一化(1/255) + 转为NCHW + BGR->RGB
    cv::Mat tensor = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(), true, false);
    */
    //----------YOLOlettlebox预处理阶段----------
    float scale = std::min(static_cast<float>(this->input_w) / frame.cols,static_cast<float>(this->input_h) / frame.rows);
    int new_w = static_cast<int>(frame.cols * scale);
    int new_h = static_cast<int>(frame.rows * scale);
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas = cv::Mat::zeros(cv::Size(this->input_w, this->input_h), CV_8UC3);
    int left = (this->input_w - new_w) / 2;
    int top = (this->input_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));
    cv::Mat tensor = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);
    

    size_t tpixels = 3 * this->input_h * this->input_w;
    std::array<int64_t, 4>input_shape_info{ 1,3,this->input_h,this->input_w };
    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); //创建 CPU 内存信息对象，用于告诉 ONNX Runtime 张量数据存储在 CPU 的哪个区域。
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, tensor.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

    std::vector<Ort::Value> outputs;
    try {
        outputs = this->session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, inputNames.size(), outputNames.data(), outputNames.size());
    }
    catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return;
    }

    // -------------------- 后处理 --------------------

    const float* raw_output = outputs[0].GetTensorMutableData<float>();

    cv::Mat dout(this->output_h, this->output_w, CV_32F, (float*)raw_output);
    cv::Mat det_output = dout.t();

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    for (int i = 0; i < det_output.rows; ++i) {
        // 类别分数从第5列开始
        cv::Mat scores = det_output.row(i).colRange(4, this->output_h);
        double max_score;
        cv::Point max_loc;
        cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &max_loc);
        int class_id = max_loc.x;
        if (max_score > 0.25) {
            //lettlebox后处理
            
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5f * ow - left ) / scale);
            int y = static_cast<int>((cy - 0.5f * oh - top ) / scale);
            int width = static_cast<int>(ow / scale);
            int height = static_cast<int>(oh / scale);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;
            
            /* 左上角黑色填充后处理
            int x = static_cast<int>((cx - 0.5 * ow) * xy_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * xy_factor);
            int width = static_cast<int>(ow * xy_factor);
            int height = static_cast<int>(oh * xy_factor);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;
            */

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

YOLODetector::~YOLODetector() {
    this->session.release();
    this->env.release();
    this->session_options.release();
    std::cout << "destory instance done" << endl;
}
