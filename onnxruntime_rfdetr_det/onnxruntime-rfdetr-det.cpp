#include "rfdetr-det.h"
void YOLODetector::initConfig(std::string onnxpath, std::string textpath) {
    // 1. 读取类别文件
    std::ifstream fp(textpath);
    std::string name;
    while (std::getline(fp, name)) {
        if (!name.empty()) {
            this->labels.push_back(name);
        }
    }
    fp.close();

    this->session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    //OrtStatusPtr status1 = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    //OrtStatusPtr status1 = OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0);
    OrtStatusPtr status = OrtSessionOptionsAppendExecutionProvider_CUDA(this->session_options, 0); //为 session_options 添加 CUDA 执行提供者（即使用 GPU 推理）
    std::wstring wpath(onnxpath.begin(), onnxpath.end());
    this->session = Ort::Session(this->env, wpath.c_str(), this->session_options);

    // 5. 获取输入输出节点名称（假设各只有一个）
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_ptr = this->session.GetInputNameAllocated(0, allocator); //输入名称和对应索引
    this->input_node_names.push_back(input_name_ptr.get());

    Ort::AllocatedStringPtr output_det_ptr = this->session.GetOutputNameAllocated(0, allocator);//输出名称和对应索引
    this->output_node_names.push_back(output_det_ptr.get());

    Ort::AllocatedStringPtr output_label_ptr = this->session.GetOutputNameAllocated(1, allocator);//输出名称和对应索引
    this->output_node_names.push_back(output_label_ptr.get());


    this->inputNames = { this->input_node_names[0].c_str() }; //准备输入名称数组,c风格
    this->outputNames = { this->output_node_names[0].c_str(),this->output_node_names[1].c_str()};//准备输出名称数组,c风格
    // 可选：打印输入输出形状
    std::vector<int64_t> input_shape = this->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Input shape: " << input_shape[1] << "x" << input_shape[2] << "x" << input_shape[3] << std::endl;
    std::vector<int64_t> output_det_shape = this->session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Outputdet shape: " << output_det_shape[1] << "x" << output_det_shape[2] << std::endl;
    this->output_det_h = output_det_shape[1];
    this->output_det_w = output_det_shape[2];

    std::vector<int64_t> output_label_shape = this->session.GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Outputdet shape: " << output_label_shape[1] << "x" << output_label_shape[2] << std::endl;
    this->output_label_h = output_label_shape[1];
    this->output_label_w = output_label_shape[2];
}

double YOLODetector::sigmoid_stable(double x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    else {
        return std::exp(x) / (1.0 + std::exp(x));
    }
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
    float xy_factor = static_cast<float>(_max) / static_cast<float>(this->input_w);
    // blobFromImage: 缩放 + 归一化(1/255) + 转为NCHW + BGR->RGB
    
    cv::Mat resizedimage;
    cv::resize(canvas, resizedimage, cv::Size(576, 576));
    resizedimage.convertTo(resizedimage, CV_32F, 1.0 / 255.0);

    cv::Mat meanImg = cv::Mat::zeros(resizedimage.size(), resizedimage.type());
    meanImg.setTo(cv::Scalar(0.485, 0.456, 0.406));

    cv::Mat stdImg = cv::Mat::zeros(resizedimage.size(), resizedimage.type());
    stdImg.setTo(cv::Scalar(0.229, 0.224, 0.225));

    cv::subtract(resizedimage, meanImg, resizedimage);
    cv::divide(resizedimage, stdImg, resizedimage);

    cv::Mat tensor = cv::dnn::blobFromImage(resizedimage);

    size_t tpixels = 3 * this->input_h * this->input_w;
    std::array<int64_t, 4>input_shape_info{ 1,3,this->input_h,this->input_w };
    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); //创建 CPU 内存信息对象，用于告诉 ONNX Runtime 张量数据存储在 CPU 的哪个区域。
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, tensor.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

    std::vector<Ort::Value> outputs;
    try {
        outputs = this->session.Run(Ort::RunOptions{ nullptr }, this->inputNames.data(), &input_tensor_, this->inputNames.size(), this->outputNames.data(), this->outputNames.size());
    }
    catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return;
    }

    // -------------------- 后处理 --------------------

    const float* raw_output1 = outputs[0].GetTensorMutableData<float>();
    cv::Mat det_output(this->output_det_h, this->output_det_w, CV_32F, (float*)raw_output1);

    const float* raw_output2 = outputs[1].GetTensorMutableData<float>();
    cv::Mat label_output(this->output_label_h, this->output_label_w, CV_32F, (float*)raw_output2);

    for (int i = 0; i < det_output.rows; i++) {
        //最后一个类别是背景;
        cv::Mat classes_scores = label_output.row(i).colRange(0, label_output.cols);
        cv::Point classIdPoint;
        double maxscore;
        minMaxLoc(classes_scores, 0, &maxscore, 0, &classIdPoint);
        maxscore = this->sigmoid_stable(maxscore);
        if (maxscore > this->conf_t)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = ((cx - 0.5 * ow) * _max);
            int y = ((cy - 0.5 * oh) * _max);
            int width = (ow * _max);
            int height = (oh * _max);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;
            cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1, 8);
            cv::rectangle(frame, cv::Point(box.tl().x, box.tl().y - 20), cv::Point(box.br().x, box.tl().y), cv::Scalar(0, 255, 255), -1);
            cv::putText(frame, this->labels[classIdPoint.x], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
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
