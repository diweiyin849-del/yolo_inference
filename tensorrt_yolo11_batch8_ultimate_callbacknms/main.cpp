
#include "ultimate-nms-callback.h"
#include <chrono>
#include <map>
#include <atomic>
#include <queue>

std::mutex display_mutex;
std::queue<std::pair<int, std::vector<std::vector<DetectResult>>>> display_queue;
std::atomic<bool> display_stop{ false };
std::atomic<double> g_infer_fps{ 0.0 };   // 全局推理 FPS

int main(int argc, char** argv) {
    std::vector<std::string> classNames;
    std::ifstream fp("D:/yolo_models/yolo-cls.txt");
    std::string name;
    while (getline(fp, name)) if (!name.empty()) classNames.push_back(name);
    fp.close();

    std::unique_ptr<YOLOv81112TRTDetector> detector = std::make_unique<YOLOv81112TRTDetector>();
    detector->initConfig("D:/TensorRT-10.8.0.43/bin/yolo11n-dynamic-batch16.engine");

    cv::VideoCapture capture("D:/yolo_models/play_scoers.mp4");
    if (!capture.isOpened()) {
        std::cerr << "Failed to open video!" << std::endl;
        return -1;
    }

    const int BATCH_SIZE = 16;
    std::vector<cv::Mat> frame_buffer;
    frame_buffer.reserve(BATCH_SIZE);
    int frame_index = 0;
    std::map<int, cv::Mat> frame_store;

    // 推理 FPS 统计变量
    auto infer_last_time = std::chrono::steady_clock::now();
    int infer_count = 0;

    // 显示线程：只显示最新帧
    std::thread display_thread([&]() {
        auto display_last_time = std::chrono::steady_clock::now();
        double display_fps = 0.0;
        int display_count = 0;

        while (!display_stop) {
            std::pair<int, std::vector<std::vector<DetectResult>>> item;
            {
                std::lock_guard<std::mutex> lock(display_mutex);
                if (display_queue.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                // 丢弃旧批，只保留最新一批
                while (display_queue.size() > 1) {
                    display_queue.pop();
                }
                item = std::move(display_queue.front());
                display_queue.pop();
            }
            int batch_start = item.first;
            auto& results = item.second;
            for (int b = 0; b < (int)results.size(); ++b) {
                int frame_id = batch_start + b;
                auto it = frame_store.find(frame_id);
                if (it != frame_store.end()) {
                    cv::Mat show = it->second.clone();
                    for (const auto& dr : results[b]) {
                        cv::rectangle(show, dr.box, cv::Scalar(0, 0, 255), 2);
                        cv::putText(show, classNames[dr.classId],
                            cv::Point(dr.box.x, dr.box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                    }

                    // 显示 FPS
                    auto now = std::chrono::steady_clock::now();
                    double delta = std::chrono::duration<double>(now - display_last_time).count();
                    display_last_time = now;
                    if (delta > 0) {
                        double fps = 1.0 / delta;
                        display_fps = (display_count == 0) ? fps : 0.9 * display_fps + 0.1 * fps;
                        display_count++;
                    }

                    char txt[128];
                    snprintf(txt, sizeof(txt), "Display: %.1f FPS", display_fps);
                    cv::putText(show, txt, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

                    // 显示推理 FPS（从原子变量读取）
                    double infer_fps = g_infer_fps.load();
                    snprintf(txt, sizeof(txt), "Inference: %.1f FPS", infer_fps);
                    cv::putText(show, txt, cv::Point(show.cols - 220, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

                    cv::imshow("Async PostProcess", show);
                    cv::waitKey(1);
                    frame_store.erase(it);
                }
            }
        }
        });

    // 主循环
    while (true) {
        cv::Mat frame;
        capture >> frame;
        if (frame.empty()) break;
        frame_store[frame_index] = frame.clone();
        frame_buffer.push_back(frame.clone());
        frame_index++;

        if ((int)frame_buffer.size() == BATCH_SIZE) {
            int batch_start = frame_index - BATCH_SIZE;
            std::vector<std::vector<DetectResult>> dummy;

            // 计算推理 FPS（吞吐量）
            auto now = std::chrono::steady_clock::now();
            double delta = std::chrono::duration<double>(now - infer_last_time).count();
            infer_last_time = now;
            if (delta > 0) {
                double fps = BATCH_SIZE / delta;
                g_infer_fps = (infer_count == 0) ? fps : 0.9 * g_infer_fps.load() + 0.1 * fps;
                infer_count++;
            }

            detector->detectBatch(frame_buffer, dummy, [batch_start](const std::vector<std::vector<DetectResult>>& results) {
                std::lock_guard<std::mutex> lock(display_mutex);
                display_queue.push({ batch_start, results });
                });
            frame_buffer.clear();
            frame_buffer.reserve(BATCH_SIZE);
        }
    }

    if (!frame_buffer.empty()) {
        std::cerr << "Warning: leftover " << frame_buffer.size() << " frames dropped." << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    display_stop = true;
    if (display_thread.joinable()) display_thread.join();

    capture.release();
    cv::destroyAllWindows();

    std::cout << "Final Inference FPS: " << g_infer_fps.load() << std::endl;
    return 0;
}