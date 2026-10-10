// yolo11-det.cpp
#include "yolo11-batch8-det.h"

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

YOLOv81112TRTDetector::~YOLOv81112TRTDetector() {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    if (buffers[0]) { cudaFree(buffers[0]); buffers[0] = nullptr; }
    if (buffers[1]) { cudaFree(buffers[1]); buffers[1] = nullptr; }
    std::cout << "destory instance done!" << std::endl;
}

void YOLOv81112TRTDetector::initConfig(std::string enginepath) {
   

    // 1. 读取 engine 文件
    std::ifstream file(enginepath, std::ios::binary);
    std::vector<char> trtModelStream;
    int size = 0;
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream.resize(size);
        file.read(trtModelStream.data(), size);
        file.close();
    }

    // 2. 创建 runtime / engine / context
    this->runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    this->engine = std::unique_ptr<ICudaEngine>(this->runtime->deserializeCudaEngine(trtModelStream.data(), size));
    this->context = std::unique_ptr<IExecutionContext>(this->engine->createExecutionContext());

    // 3. 获取输入输出维度（包含 batch）
    auto input_shape = engine->getTensorShape("images");
    auto output_shape = engine->getTensorShape("output0");

    this->m_batchSize = input_shape.d[0];   // 例如 8
    this->input_h = input_shape.d[2];
    this->input_w = input_shape.d[3];
    this->output_h = output_shape.d[1];     // 每帧的通道数（如 84）
    this->output_w = output_shape.d[2];     // 每帧的检测框数（如 8400）

    printf("batch: %d, inputH: %d, inputW: %d \n", m_batchSize, input_h, input_w);
    std::cout << "out data format: " << output_h << "x" << output_w << std::endl;

    // 4. 分配 GPU 内存（按 batch 大小）
    size_t input_byte = m_batchSize * input_h * input_w * 3 * sizeof(float);
    size_t output_byte = m_batchSize * output_h * output_w * sizeof(float);
    cudaMalloc(&buffers[0], input_byte);
    cudaMalloc(&buffers[1], output_byte);

    // 5. CPU 端输出缓存（同样按 batch 大小）
    this->prob = cv::Mat(Size(output_w, output_h * m_batchSize), CV_32F);

    // 6. 创建 CUDA 流
    cudaStreamCreate(&stream);
}

// ---------------- 单帧推理（保留，但不再使用） ----------------
//void YOLOv81112TRTDetector::detect(cv::Mat& frame, std::vector<DetectResult>& results) {
//    // 这里保留原有单帧逻辑，但建议使用 detectBatch
//    // 为了简洁，此处略去实现（你可以保留旧代码，或直接调用 detectBatch 包装）
//    // 我们将在主程序中使用 detectBatch，所以此方法可以留空或调用 detectBatch
//    // 但为了完整性，提供一个简单的实现（但实际不会用到）
//    std::vector<cv::Mat> frames = { frame };
//    std::vector<std::vector<DetectResult>> batch_results;
//    detectBatch(frames, batch_results);
//    if (!batch_results.empty())
//        results = batch_results[0];
//}
//处理单帧
void YOLOv81112TRTDetector::detect(cv::Mat& frame, std::vector<DetectResult>& results) {
    int64 start = cv::getTickCount();
    // 图象预处理 - 格式化操作
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    //cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    //cv::Rect roi(0, 0, w, h);
    //frame.copyTo(image(roi));
    //float x_factor = image.cols / (float)this->input_w;
    //float y_factor = image.rows / (float)this->input_h;
    //cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0), true, false);
    //// 内存到显存,GPU目标地址, CPU源地址,方向: CPU→GPU,异步流
    //cudaMemcpyAsync(buffers[0], blob.ptr<float>(), 3 * this->input_h * this->input_w * sizeof(float), cudaMemcpyHostToDevice, stream);
    //// GPU推理
    // 图象预处理 - 格式化操作
    cv::cuda::GpuMat gpu_frame;
    gpu_frame.upload(frame);
    cv::cuda::GpuMat gpu_image = cv::cuda::GpuMat(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    gpu_frame.copyTo(gpu_image(roi));
    float x_factor = gpu_image.cols / static_cast<float>(this->input_w);
    float y_factor = gpu_image.rows / static_cast<float>(this->input_h);
    //Resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_image, resized, cv::Size(this->input_w, this->input_h), 0, 0, cv::INTER_LINEAR);
    //To float32 (0-1)
    cv::cuda::GpuMat float_image;
    resized.convertTo(float_image, CV_32FC3, 1.f / 255.f);
    // BGR To RGB
    cv::cuda::GpuMat rgb;
    cv::cuda::cvtColor(float_image, rgb, cv::COLOR_BGR2RGB);
    std::vector<cv::cuda::GpuMat> rgb_out;
    //cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    //表示第 i 个 GpuMat 的起始地址指向 buffers[0] 偏移 i * input_w * input_h 个 float 元素的位置。
    //用户提供的 GPU 显存指针，GpuMat 不会自己分配显存，而是直接使用指定的 GPU 显存
    for (size_t i = 0; i < 3; ++i)
    {
        rgb_out.emplace_back(cv::cuda::GpuMat(cv::Size(this->input_w, this->input_h), CV_32FC1, (float*)buffers[0] + i * this->input_w * this->input_h));
    }
    // opencv HWC order -> CHW order
    cv::cuda::split(rgb, rgb_out);
    gpu_frame.release();
    resized.release();
    float_image.release();
    rgb.release();
    gpu_image.release();
    this->context->executeV2(buffers);
    // 显存到内存 blob.data：返回 uchar*，最底层的字节指针 blob.ptr<float>()：返回 float*，类型安全的float指针
    cudaMemcpyAsync(prob.ptr<float>(), buffers[1], this->output_h * this->output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // 后处理, 1x25200x85
    std::vector<Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    cv::Mat dout(this->output_h, this->output_w, CV_32F, prob.ptr<float>());  //格式化
    cv::Mat det_output = dout.t(); // 转置变成8400x84
    for (int i = 0; i < det_output.rows; i++) {
        cv::Mat classes_scores = det_output.row(i).colRange(4, this->output_h); //output_h=84,对于单行矩阵，minMaxLoc 返回的 Point 中：x 坐标表示最大值所在的列索引（即类别 ID）y 坐标始终为 0
        cv::Point classIdPoint;
        double score;
        minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
        // 置信度 0～1之间
        if (score > this->score_threshold)
        {
            float cx = det_output.at<float>(i, 0);   //输出中心点坐标和宽高；
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = ((cx - 0.5 * ow) * x_factor);
            int y = ((cy - 0.5 * oh) * y_factor);
            int width = (ow * x_factor);
            int height = (oh * y_factor);
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
        DetectResult dr;
        int index = indexes[i];
        int idx = classIds[index];
        dr.box = boxes[index];
        dr.classId = idx;
        dr.score = confidences[index];
        results.push_back(dr);
    }
    // 计算FPS render it
    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2, 8);
}


//处理多帧
void YOLOv81112TRTDetector::detectBatch(const std::vector<cv::Mat>& frames,std::vector<std::vector<DetectResult>>& batch_results) {
    int64 start = cv::getTickCount();
    int num_frames = frames.size();
    if (num_frames != m_batchSize) {
        std::cerr << "Warning: frame count (" << num_frames<< ") != batch size (" << m_batchSize << "). " << std::endl;
        // 若帧数不足，补零或直接返回，这里简单处理为仅处理有效帧
        if (num_frames < m_batchSize) {
            std::cerr << "Error: not enough frames, abort." << std::endl;
            return;
        }
        // 若多于 batch，只取前 m_batchSize 帧，但这里安全返回
        return;
    }

    // 1. 批量batch的cpu预处理：letterbox + 归一化
    //std::vector<float> x_factors(m_batchSize), y_factors(m_batchSize);

    //for (int b = 0; b < m_batchSize; ++b) {
    //    cv::Mat frame = frames[b];
    //    int w = frame.cols;
    //    int h = frame.rows;
    //    int _max = std::max(h, w);
    //    // 创建画布
    //    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    //    frame.copyTo(image(cv::Rect(0, 0, w, h)));
    //    // 缩放至模型输入尺寸
    //    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
    //    cudaMemcpyAsync(static_cast<char*>(buffers[0]) + b * input_h * input_w * 3 *sizeof(float), blob.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    //    // 计算坐标缩放因子（画布 → 原图）
    //    x_factors[b] = image.cols / (float)this->input_w;
    //    y_factors[b] = image.rows / (float)this->input_h;
    //}

    // 2 批量batch的Opencv-GPU预处理：letterbox + 归一化
    //float* base = static_cast<float*>(buffers[0]);
    //std::vector<float> x_factors(m_batchSize), y_factors(m_batchSize);

    //for (int b = 0; b < m_batchSize; ++b) {
    //    cv::Mat frame = frames[b];
    //    int w = frame.cols, h = frame.rows;
    //    int _max = std::max(h, w);

    //    cv::cuda::GpuMat gpu_frame;
    //    gpu_frame.upload(frame);

    //    cv::cuda::GpuMat gpu_image(cv::Size(_max, _max), CV_8UC3);
    //    cv::Rect roi(0, 0, w, h);
    //    gpu_frame.copyTo(gpu_image(roi));

    //    cv::cuda::GpuMat resized;
    //    cv::cuda::resize(gpu_image, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

    //    cv::cuda::GpuMat float_image;
    //    resized.convertTo(float_image, CV_32FC3, 1.f / 255.f);

    //    cv::cuda::GpuMat rgb;
    //    cv::cuda::cvtColor(float_image, rgb, cv::COLOR_BGR2RGB);

    //    // 正确的偏移写入
    //    size_t offset = b * 3 * input_w * input_h;
    //    std::vector<cv::cuda::GpuMat> rgb_out;
    //    for (int c = 0; c < 3; ++c) {
    //        rgb_out.emplace_back(cv::cuda::GpuMat(cv::Size(input_w, input_h),CV_32FC1,base + offset + c * input_w * input_h));
    //    }
    //    cv::cuda::split(rgb, rgb_out);

    //    // 释放临时对象
    //    gpu_frame.release();
    //    gpu_image.release();
    //    resized.release();
    //    float_image.release();
    //    rgb.release();

    //    // 缩放因子
    //    x_factors[b] = _max / (float)input_w;
    //    y_factors[b] = _max / (float)input_h;
    //}

    //批量batch的Opencv-GPU预处理+CUDAstream：letterbox + 归一化
    cv::cuda::Stream streamtemp;  // ? 创建流

    float* base = static_cast<float*>(buffers[0]);
    std::vector<float> x_factors(m_batchSize), y_factors(m_batchSize);

    for (int b = 0; b < m_batchSize; ++b) {
        cv::Mat frame = frames[b];
        int w = frame.cols, h = frame.rows;
        int _max = std::max(h, w);

        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame, streamtemp);          // ? 加 stream

        cv::cuda::GpuMat gpu_image(cv::Size(_max, _max), CV_8UC3);
        cv::Rect roi(0, 0, w, h);
        gpu_frame.copyTo(gpu_image(roi), streamtemp); // 加 stream

        cv::cuda::GpuMat resized;
        cv::cuda::resize(gpu_image, resized, cv::Size(input_w, input_h),0, 0, cv::INTER_LINEAR, streamtemp);       // 加 stream

        cv::cuda::GpuMat float_image;
        resized.convertTo(float_image, CV_32FC3, 1.f / 255.f, streamtemp); // 加 stream

        cv::cuda::GpuMat rgb;
        cv::cuda::cvtColor(float_image, rgb, cv::COLOR_BGR2RGB, 0, streamtemp);
        // 正确偏移写入
        size_t offset = b * 3 * input_w * input_h;
        std::vector<cv::cuda::GpuMat> rgb_out;
        for (int c = 0; c < 3; ++c) {
            rgb_out.emplace_back(cv::cuda::GpuMat(
                cv::Size(input_w, input_h), CV_32FC1,
                base + offset + c * input_w * input_h
            ));
        }
        cv::cuda::split(rgb, rgb_out, streamtemp);   // 加 stream

        // 释放临时对象（release 无 stream 参数，它只修改引用计数，瞬间完成）
        gpu_frame.release();
        gpu_image.release();
        resized.release();
        float_image.release();
        rgb.release();

        // 缩放因子（纯 CPU 计算，直接赋值）
        x_factors[b] = _max / (float)input_w;
        y_factors[b] = _max / (float)input_h;
    }

    //  等待所有预处理操作完成（必须！）
    streamtemp.waitForCompletion();
    // 3. 推理
    this->context->executeV2(buffers);

    // 4. 拷贝输出

    cudaMemcpyAsync(prob.ptr<float>(), buffers[1], m_batchSize * output_h * output_w * sizeof(float), cudaMemcpyDeviceToHost, this->stream);
    cudaStreamSynchronize(this->stream);

    // 5. 后处理每一帧
    for (int b = 0; b < m_batchSize; ++b) {
        cv::Mat dout(output_h, output_w, CV_32F, prob.ptr<float>() + b * output_h * output_w);
        cv::Mat det_output = dout.t(); // 转置为 (8400, 84)
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;

        for (int i = 0; i < det_output.rows; ++i) {
            cv::Mat classes_scores = det_output.row(i).colRange(4, output_h);
            cv::Point classIdPoint;
            double score;
            cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
            if (score > score_threshold) {
                float cx = det_output.at<float>(i, 0);
                float cy = det_output.at<float>(i, 1);
                float ow = det_output.at<float>(i, 2);
                float oh = det_output.at<float>(i, 3);
                // 使用对应帧的缩放因子
                float x_factor = x_factors[b];
                float y_factor = y_factors[b];
                int x = static_cast<int>((cx - 0.5f * ow) * x_factor);
                int y = static_cast<int>((cy - 0.5f * oh) * y_factor);
                int width = static_cast<int>(ow * x_factor);
                int height = static_cast<int>(oh * y_factor);
                boxes.push_back(cv::Rect(x, y, width, height));
                classIds.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(score));
            }
        }
        // NMS
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);

        std::vector<DetectResult> results;
        for (int idx : indexes) {
            DetectResult dr;
            dr.classId = classIds[idx];
            dr.score = confidences[idx];
            dr.box = boxes[idx];
            results.push_back(dr);
        }
        batch_results.push_back(results);
    }
}