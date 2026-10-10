# YOLO Inference

基于 C++ 的计算机视觉模型推理示例集合，使用 Visual Studio 工程演示不同推理后端下的模型加载、图像预处理、推理、后处理与结果可视化。

## 主要内容

- 推理后端：ONNX Runtime、OpenCV DNN、OpenVINO、TensorRT。
- 模型示例：YOLOv5、YOLO11、YOLO26、RT-DETR、RF-DETR（具体组合以各子工程为准）。
- 任务类型：图像分类、目标检测、旋转框检测、姿态估计、实例分割和深度估计。
- 进阶示例：OpenVINO 异步推理、队列与回调，以及 TensorRT 批量推理。

## 项目结构

| 目录 / 文件 | 说明 |
| --- | --- |
| `yolo_item.sln` | Visual Studio 解决方案入口 |
| `onnxruntime_*` | ONNX Runtime 推理示例 |
| `opencvdnn_*` | OpenCV DNN 推理示例 |
| `openvino_*` / `Openvino_*` | OpenVINO 推理示例 |
| `tensorrt_*` / `Tensorrt_*` | TensorRT 推理示例 |

多数子工程包含 `main.cpp`、模型推理实现与对应头文件，可单独阅读和运行。

## 使用方法

1. 使用 Visual Studio 2022 打开 `yolo_item.sln`，安装“使用 C++ 的桌面开发”工作负载与 Windows SDK。
2. 准备 OpenCV，以及所选子工程需要的 ONNX Runtime、OpenVINO 或 TensorRT。GPU 示例还需要相应 CUDA / cuDNN 环境。
3. 在子工程属性中，将包含目录、库目录和链接依赖修改为本机 SDK 路径。现有工程配置包含作者本机绝对路径，需要自行调整。
4. 准备相应模型权重、类别标签以及图片或视频，并修改子工程 `main.cpp` 中的资源路径。模型、标签与视频资源不随源码提供。
5. 将目标子工程设为启动项目，优先选择 `Release | x64` 编译运行，并确保依赖 DLL 可被程序找到。

例如，`onnxruntime_yolo11_det/main.cpp` 使用 YOLO11 ONNX 模型对视频逐帧检测，并通过 OpenCV 显示结果。

## 说明

本项目用于学习和参考不同后端的 C++ 推理流程。各示例依赖与模型输出格式可能不同，请以对应源码为准。仓库仅包含源码和工程文件，编译产物、IDE 缓存、运行时二进制与模型文件不纳入版本控制。
