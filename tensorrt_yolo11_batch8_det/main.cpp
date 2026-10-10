#include "yolo11-batch8-det.h"

int main(int argc, char** argv) {
    std::vector<std::string> classNames;
    std::string label_map = "D:/yolo_models/yolo-cls.txt";
    // 1. 读取类别名
    std::ifstream fp(label_map);
    std::string name;
    while (getline(fp, name)) {
        if (!name.empty()) {
            classNames.push_back(name);
        }
    }
    fp.close();
    // 2. 初始化检测器
    std::unique_ptr<YOLOv81112TRTDetector> detector = std::make_unique<YOLOv81112TRTDetector>();
    detector->initConfig("D:/TensorRT-10.8.0.43/bin/yolo11n-dynamic-batch16.engine");

    // 3. 打开视频
    cv::VideoCapture capture("D:/yolo_models/play_scoers.mp4");
    if (!capture.isOpened()) {
        std::cerr << "Failed to open video!" << std::endl;
        return -1;
    }

    const int BATCH_SIZE = 16;   // 应与 engine 的 batch 一致
    std::vector<cv::Mat> frame_buffer;
    frame_buffer.reserve(BATCH_SIZE);

    while (true) {
        cv::Mat frame;
        capture.read(frame);
        if (frame.empty()) break;
        frame_buffer.push_back(frame);
        // 每凑够 BATCH_SIZE 帧，进行一次批量推理
        if (frame_buffer.size() == BATCH_SIZE) {
            auto start_batch = std::chrono::steady_clock::now();
            std::vector<std::vector<DetectResult>> batch_results;
            detector->detectBatch(frame_buffer, batch_results);
            auto end_batch = std::chrono::steady_clock::now();
            double batch_time = std::chrono::duration<double>(end_batch - start_batch).count();
            double fps = BATCH_SIZE / batch_time;
            for (int b = 0; b < BATCH_SIZE; ++b) {
                cv::Mat& show_frame = frame_buffer[b];
                for (const auto& dr : batch_results[b]) {
                    cv::rectangle(show_frame, dr.box, cv::Scalar(0, 0, 255), 1);
                    cv::putText(show_frame, classNames[dr.classId],cv::Point(dr.box.x, dr.box.y - 10),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
                }
                // 绘制 FPS
                cv::putText(show_frame, cv::format("FPS: %.2f", fps),cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2, 8);
                cv::imshow("YOLO11n Batch=" + std::to_string(BATCH_SIZE), show_frame);
            }
            frame_buffer.clear();
            if (cv::waitKey(1) == 27) break;
        }
    }

    // 处理不足 BATCH_SIZE 的剩余帧（简单处理：逐帧推理）
    if (!frame_buffer.empty()) {
        // 可以补零或单独处理，这里简单逐帧调用 detect
        for (auto& f : frame_buffer) {
            std::vector<DetectResult> results;
            detector->detect(f, results);
            for (auto& dr : results) {
                cv::rectangle(f, dr.box, cv::Scalar(0, 0, 255), 1);
                cv::putText(f, classNames[dr.classId], cv::Point(dr.box.x, dr.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
            }
            cv::imshow("remaining", f);
            cv::waitKey(1);
        }
    }

    capture.release();
    cv::destroyAllWindows();
    return 0;
}