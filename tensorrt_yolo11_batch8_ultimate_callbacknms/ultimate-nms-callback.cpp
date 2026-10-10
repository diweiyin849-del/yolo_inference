
#include "ultimate-nms-callback.h"

static Logger gLogger;

YOLOv81112TRTDetector::YOLOv81112TRTDetector() {}

YOLOv81112TRTDetector::~YOLOv81112TRTDetector() {
    {
        std::lock_guard<std::mutex> lock(post_mutex_);
        stop_post_ = true;
    }
    post_cv_.notify_all();
    if (post_thread_.joinable())
        post_thread_.join();

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    if (buffers[0]) { cudaFree(buffers[0]); buffers[0] = nullptr; }
    if (buffers[1]) { cudaFree(buffers[1]); buffers[1] = nullptr; }
    std::cout << "destroy instance done!" << std::endl;
}

void YOLOv81112TRTDetector::initConfig(std::string enginepath) {
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

    this->runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    this->engine = std::unique_ptr<ICudaEngine>(this->runtime->deserializeCudaEngine(trtModelStream.data(), size));
    this->context = std::unique_ptr<IExecutionContext>(this->engine->createExecutionContext());

    auto input_shape = engine->getTensorShape("images");
    auto output_shape = engine->getTensorShape("output0");
    this->m_batchSize = input_shape.d[0];
    this->input_h = input_shape.d[2];
    this->input_w = input_shape.d[3];
    this->output_h = output_shape.d[1];
    this->output_w = output_shape.d[2];

    printf("batch: %d, inputH: %d, inputW: %d \n", m_batchSize, input_h, input_w);
    std::cout << "out data format: " << output_h << "x" << output_w << std::endl;

    size_t input_byte = m_batchSize * input_h * input_w * 3 * sizeof(float);
    size_t output_byte = m_batchSize * output_h * output_w * sizeof(float);
    cudaMalloc(&buffers[0], input_byte);
    cudaMalloc(&buffers[1], output_byte);

    this->prob = cv::Mat(Size(output_w, output_h * m_batchSize), CV_32F);

    cudaStreamCreate(&stream);
    const char* input_name = "images";
    const char* output_name = "output0";
    this->context->setTensorAddress(input_name, buffers[0]);
    this->context->setTensorAddress(output_name, buffers[1]);

    post_thread_ = std::thread(&YOLOv81112TRTDetector::postProcessingWorker, this);
}

void YOLOv81112TRTDetector::detect(cv::Mat& frame, std::vector<DetectResult>& results) {
    // 单帧推理（保持不变）
}

void YOLOv81112TRTDetector::postProcessingWorker() {
    while (true) {
        PostTask task;
        {
            std::unique_lock<std::mutex> lock(post_mutex_);
            post_cv_.wait(lock, [this] { return !post_queue_.empty() || stop_post_; });
            if (stop_post_ && post_queue_.empty())
                break;
            task = std::move(post_queue_.front());
            post_queue_.pop();
        }
        // 等待 GPU 拷贝完成（数据已就绪）
        cudaStreamSynchronize(task.stream);
        runPostProcessing(task);
    }
}

void YOLOv81112TRTDetector::runPostProcessing(const PostTask& task) {
    std::vector<std::vector<DetectResult>> batch_results;
    batch_results.reserve(task.batch_size);

    for (int b = 0; b < task.batch_size; ++b) {
        cv::Mat dout(task.output_h, task.output_w, CV_32FC1,
            (void*)(task.prob_data.ptr<float>() + b * task.output_h * task.output_w));
        cv::Mat det_output = dout.t();
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;

        for (int i = 0; i < det_output.rows; ++i) {
            cv::Mat classes_scores = det_output.row(i).colRange(4, task.output_h);
            cv::Point classIdPoint;
            double score;
            cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
            if (score > score_threshold) {
                float cx = det_output.at<float>(i, 0);
                float cy = det_output.at<float>(i, 1);
                float ow = det_output.at<float>(i, 2);
                float oh = det_output.at<float>(i, 3);
                float x_factor = task.x_factors[b];
                float y_factor = task.y_factors[b];
                int x = static_cast<int>((cx - 0.5f * ow) * x_factor);
                int y = static_cast<int>((cy - 0.5f * oh) * y_factor);
                int width = static_cast<int>(ow * x_factor);
                int height = static_cast<int>(oh * y_factor);
                boxes.push_back(cv::Rect(x, y, width, height));
                classIds.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(score));
            }
        }
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);

        std::vector<DetectResult> results;
        results.reserve(indexes.size());
        for (int idx : indexes) {
            DetectResult dr;
            dr.classId = classIds[idx];
            dr.score = confidences[idx];
            dr.box = boxes[idx];
            results.push_back(dr);
        }
        batch_results.push_back(std::move(results));
    }

    if (task.callback) {
        task.callback(batch_results);
    }
}

void YOLOv81112TRTDetector::syncPostProcessing(const cv::Mat& prob_data,
    const std::vector<float>& x_factors,
    const std::vector<float>& y_factors,
    std::vector<std::vector<DetectResult>>& results) {
    PostTask task;
    task.prob_data = prob_data.clone();
    task.x_factors = x_factors;
    task.y_factors = y_factors;
    task.callback = [&results](const std::vector<std::vector<DetectResult>>& res) {
        results = res;
        };
    task.output_h = this->output_h;
    task.output_w = this->output_w;
    task.batch_size = this->m_batchSize;
    task.stream = this->stream;
    runPostProcessing(task);
}

// ---------- 批量推理（核心改动：异步分支分配新内存，主线程不等待） ----------
void YOLOv81112TRTDetector::detectBatch(const std::vector<cv::Mat>& frames,
    std::vector<std::vector<DetectResult>>& batch_results,
    PostProcessCallback callback) {
    int num_frames = frames.size();
    if (num_frames != m_batchSize) {
        std::cerr << "Error: frame count " << num_frames << " != batch size " << m_batchSize << std::endl;
        return;
    }

    // ===== 预处理（异步） =====
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(this->stream);
    float* base = static_cast<float*>(buffers[0]);
    std::vector<float> x_factors(m_batchSize), y_factors(m_batchSize);

    for (int b = 0; b < m_batchSize; ++b) {
        cv::Mat frame = frames[b];
        int w = frame.cols, h = frame.rows;
        int _max = std::max(h, w);

        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame, cv_stream);

        cv::cuda::GpuMat gpu_image(cv::Size(_max, _max), CV_8UC3);
        cv::Rect roi(0, 0, w, h);
        gpu_frame.copyTo(gpu_image(roi), cv_stream);

        cv::cuda::GpuMat resized;
        cv::cuda::resize(gpu_image, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR, cv_stream);

        cv::cuda::GpuMat float_image;
        resized.convertTo(float_image, CV_32FC3, 1.f / 255.f, 0.0, cv_stream);

        cv::cuda::GpuMat rgb;
        cv::cuda::cvtColor(float_image, rgb, cv::COLOR_BGR2RGB, 0, cv_stream);

        size_t offset = b * 3 * input_w * input_h;
        std::vector<cv::cuda::GpuMat> rgb_out;
        for (int c = 0; c < 3; ++c) {
            rgb_out.emplace_back(cv::cuda::GpuMat(
                cv::Size(input_w, input_h), CV_32FC1,
                base + offset + c * input_w * input_h
            ));
        }
        cv::cuda::split(rgb, rgb_out, cv_stream);

        gpu_frame.release();
        gpu_image.release();
        resized.release();
        float_image.release();
        rgb.release();

        x_factors[b] = _max / (float)input_w;
        y_factors[b] = _max / (float)input_h;
    }

    // ===== 推理 =====
    if (!this->context->enqueueV3(this->stream)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }

    // ===== 异步拷贝并分流 =====
    if (callback) {
        // ---------- 异步模式（主线程不等待） ----------
        // 1. 分配专用 CPU 内存（独立于 this->prob，避免拷贝阻塞主线程）
        PostTask task;
        task.prob_data = cv::Mat(m_batchSize * output_h, output_w, CV_32F);
        task.x_factors = x_factors;
        task.y_factors = y_factors;
        task.callback = callback;
        task.output_h = this->output_h;
        task.output_w = this->output_w;
        task.batch_size = this->m_batchSize;
        task.stream = this->stream;

        // 2. 异步拷贝到刚分配的专用内存（非阻塞，立即返回）
        cudaMemcpyAsync(task.prob_data.ptr<float>(), buffers[1],
            m_batchSize * output_h * output_w * sizeof(float),
            cudaMemcpyDeviceToHost, this->stream);

        // 3. 任务入队（主线程立即返回，不等待拷贝）
        {
            std::lock_guard<std::mutex> lock(post_mutex_);
            post_queue_.push(std::move(task));
            post_cv_.notify_one();
        }
        // 注意：batch_results 不会被填充，结果通过回调返回
    }
    else {
        // ---------- 同步模式（保持原样） ----------
        cudaStreamSynchronize(this->stream);
        syncPostProcessing(this->prob, x_factors, y_factors, batch_results);
    }
}
