#include "yolo26-det.h"

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
    //OrtStatusPtr status1 = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    //OrtStatusPtr status1 = OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0);
    OrtStatusPtr status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); //为 session_options 添加 CUDA 执行提供者（即使用 GPU 推理）
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
    // -------------------- 预处理 --------------------
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    // letterbox: 创建正方形画布，填充黑色
    cv::Mat canvas = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(canvas(roi));
    float xy_factor = static_cast<float>(_max) / static_cast<float>(input_w);
    // blobFromImage: 缩放 + 归一化(1/255) + 转为NCHW + BGR->RGB
    cv::Mat tensor = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false);

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
    cv::Mat det_output(this->output_h, this->output_w, CV_32F, (float*)raw_output);
    for (int i = 0; i < det_output.rows; i++) {
        float score = det_output.at<float>(i, 4);
        // 置信度 0～1之间
        if (score > this->conf_t)
        {
            float x1 = det_output.at<float>(i, 0);
            float y1 = det_output.at<float>(i, 1);
            float x2 = det_output.at<float>(i, 2);
            float y2 = det_output.at<float>(i, 3);
            int rx1 = static_cast<int>(x1 * xy_factor);
            int ry1 = static_cast<int>(y1 * xy_factor);
            int rx2 = static_cast<int>(x2 * xy_factor);
            int ry2 = static_cast<int>(y2 * xy_factor);
            int class_id = static_cast<int>(det_output.at<float>(i, 5));
            cv::Rect box(rx1, ry1, rx2 - rx1, ry2 - ry1);
            cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1, 8);
            cv::rectangle(frame, cv::Point(box.tl().x, box.tl().y - 20), cv::Point(box.br().x, box.tl().y), cv::Scalar(0, 255, 255), -1);
            cv::putText(frame, this->labels[class_id], cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
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
