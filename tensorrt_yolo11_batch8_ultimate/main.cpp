
#include "ultimate-det.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <vector>

// ------------------ 工作线程数据结构 ------------------
struct Worker {
    std::unique_ptr<YOLOv81112TRTDetector> detector;
    std::queue<std::pair<int, std::vector<cv::Mat>>> input_queue;   // <起始帧号, 帧批> 
    //作为输入缓冲区，存储待检测的图像批次。外部线程（如视频解码线程）向此队列推送数据，后台工作线程从中取出处理。
    std::queue<std::pair<int, std::vector<std::vector<DetectResult>>>> output_queue;
    //作为输出缓冲区，存放已完成的检测结果。后台线程处理完一批图像后，将结果推入此队列，供外部线程取出。
    std::mutex mu;
    //保护两个队列的并发访问。无论是生产者（外部线程推送输入/拉取输出）还是消费者（工作线程弹出输入/推送输出），在操作任一队列前都必须锁定 mu。
    std::condition_variable cv;
    //本结构中只定义了一个 cv。它既可以用来唤醒工作线程（有新输入来了），也可以用来唤醒主线程
    std::atomic<bool> stop{ false };
    //工作线程的循环条件是 while(!stop)。当外部主线程想要销毁 Worker 时，会设置 stop = true。
    std::thread thread;
    //它代表一个实际的操作系统内核线程.这个线程会循环执行“取图 -> 推理 -> 放结果”的逻辑。
    ~Worker() {
        if (thread.joinable()) thread.join();
    }
};

// ------------------ 全局参数 ------------------
const int NUM_WORKERS = 16;          // 可调：1, 2, 4（建议先试1）
const int BATCH_SIZE = 16;          // 必须与 engine 一致

int main(int argc, char** argv) {
    // 1. 读取类别名
    std::vector<std::string> classNames;
    std::string label_map = "D:/yolo_models/yolo-cls.txt";
    std::ifstream fp(label_map);
    std::string name;
    while (getline(fp, name)) if (!name.empty()) classNames.push_back(name);
    fp.close();

    // 2. 创建工作线程（每个线程独立加载引擎）
    std::string engine_path = "D:/TensorRT-10.8.0.43/bin/yolo11n-dynamic-batch16.engine";
    std::vector<Worker> workers(NUM_WORKERS);
    for (int i = 0; i < NUM_WORKERS; ++i) {
        workers[i].detector = std::make_unique<YOLOv81112TRTDetector>();
        workers[i].detector->initConfig(engine_path);
        workers[i].thread = std::thread([&worker = workers[i]]() {
            while (true) {
                std::pair<int, std::vector<cv::Mat>> batch_pair;
                {
                    std::unique_lock<std::mutex> lock(worker.mu);
                    worker.cv.wait(lock, [&worker]() {
                        return !worker.input_queue.empty() || worker.stop.load();
                        });
                    if (worker.stop.load() && worker.input_queue.empty())
                        break;
                    batch_pair = std::move(worker.input_queue.front());
                    worker.input_queue.pop();
                }
                // 调用原始 detectBatch（不改动）
                std::vector<std::vector<DetectResult>> batch_results;
                worker.detector->detectBatch(batch_pair.second, batch_results);
                {
                    std::lock_guard<std::mutex> lock(worker.mu);
                    worker.output_queue.push({ batch_pair.first, std::move(batch_results) });
                }
            }
            });
    }

    // 3. 打开视频
    cv::VideoCapture capture("D:/yolo_models/play_scoers.mp4");
    if (!capture.isOpened()) {
        std::cerr << "Failed to open video!" << std::endl;
        return -1;
    }

    // 4. 存储容器（使用 vector，按帧序号索引）
    std::vector<cv::Mat> frame_store;                 // 存储帧副本
    std::vector<std::vector<DetectResult>> result_store; // 存储每帧的检测结果
    std::mutex store_mutex;                           // 保护 result_store 的并发写入

    // 5. 主循环变量
    std::vector<cv::Mat> frame_buffer;
    frame_buffer.reserve(BATCH_SIZE);
    int frame_index = 0;          // 当前读取的帧序号（从0开始）
    int worker_index = 0;
    int next_show_index = 0;      // 下一个要显示的帧序号

    // ---------- FPS 计算 ----------
    auto last_time = std::chrono::steady_clock::now();
    double fps = 0.0;
    int display_count = 0;

    // 6. 读取并分发
    while (true) {
        cv::Mat frame;
        capture >> frame;
        if (frame.empty()) break;

        // 存储帧副本（用于显示）
        frame_store.push_back(frame.clone());
        // 为结果预留空间（默认空）
        result_store.emplace_back(); // 初始为空

        // 将帧放入缓冲批
        frame_buffer.push_back(frame.clone());
        frame_index++;

        // 凑够一批就分发
        if ((int)frame_buffer.size() == BATCH_SIZE) {
            int batch_start = frame_index - BATCH_SIZE;
            auto& w = workers[worker_index % NUM_WORKERS];
            {
                std::lock_guard<std::mutex> lock(w.mu);
                w.input_queue.push({ batch_start, std::move(frame_buffer) });
                w.cv.notify_one();
            }
            frame_buffer.clear();
            frame_buffer.reserve(BATCH_SIZE);
            worker_index++;
        }

        // 收集所有 worker 的输出（放入 result_store）
        for (auto& w : workers) {
            std::lock_guard<std::mutex> lock(w.mu);
            while (!w.output_queue.empty()) {
                auto& item = w.output_queue.front();
                int start = item.first;
                auto& res = item.second;
                {
                    std::lock_guard<std::mutex> lock(store_mutex);
                    for (int b = 0; b < (int)res.size(); ++b) {
                        int idx = start + b;
                        if (idx < (int)result_store.size())
                            result_store[idx] = std::move(res[b]);
                    }
                }
                w.output_queue.pop();
            }
        }

        // 按序显示已完成的帧（从 next_show_index 开始检查）
        while (next_show_index < (int)result_store.size() && !result_store[next_show_index].empty()) {
            // 获取帧和结果
            cv::Mat& show_frame = frame_store[next_show_index]; // 引用，不克隆
            const auto& dets = result_store[next_show_index];

            // 绘制检测框
            for (const auto& dr : dets) {
                cv::rectangle(show_frame, dr.box, cv::Scalar(0, 0, 255), 2);
                cv::putText(show_frame, classNames[dr.classId],cv::Point(dr.box.x, dr.box.y - 5),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            }

            // ---------- 计算并绘制 FPS ----------
            auto now = std::chrono::steady_clock::now();
            double delta = std::chrono::duration<double>(now - last_time).count();
            last_time = now;
            if (delta > 0) {
                double instant_fps = 1.0 / delta;
                fps = (display_count == 0) ? instant_fps : 0.9 * fps + 0.1 * instant_fps;
                display_count++;
            }
            char fps_text[32];
            snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
            cv::putText(show_frame, fps_text,cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,cv::Scalar(0, 255, 0), 2);

            // 显示
            cv::imshow("YOLO11 Multi-thread", show_frame);
            if (cv::waitKey(1) == 27) { // ESC 退出
                capture.release();
                cv::destroyAllWindows();
                for (auto& w : workers) { w.stop = true; w.cv.notify_all(); }
                for (auto& w : workers) if (w.thread.joinable()) w.thread.join();
                return 0;
            }

            // 显示后清空结果（释放内存，可改为直接清除）
            result_store[next_show_index].clear();
            // 帧可保留，但为了节省内存，我们也可以释放
            frame_store[next_show_index].release();
            next_show_index++;
        }
    }

    // 7. 处理剩余帧（丢弃）
    if (!frame_buffer.empty()) {
        std::cerr << "Warning: leftover " << frame_buffer.size() << " frames dropped." << std::endl;
    }

    // 8. 等待所有工作线程停止
    for (auto& w : workers) {
        w.stop = true;
        w.cv.notify_all();
    }
    for (auto& w : workers) {
        if (w.thread.joinable()) w.thread.join();
    }

    // 9. 显示剩余未显示的帧（若有）
    while (next_show_index < (int)result_store.size() && !result_store[next_show_index].empty()) {
        cv::Mat& show_frame = frame_store[next_show_index];
        const auto& dets = result_store[next_show_index];
        for (const auto& dr : dets) {
            cv::rectangle(show_frame, dr.box, cv::Scalar(0, 0, 255), 2);
            cv::putText(show_frame, classNames[dr.classId],cv::Point(dr.box.x, dr.box.y - 5),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }
        cv::putText(show_frame, cv::format("FPS: %.1f", fps),cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,cv::Scalar(0, 255, 0), 2);
        cv::imshow("YOLO11 Multi-thread", show_frame);
        cv::waitKey(1);
        result_store[next_show_index].clear();
        frame_store[next_show_index].release();
        next_show_index++;
    }

    capture.release();
    cv::destroyAllWindows();
    return 0;
}