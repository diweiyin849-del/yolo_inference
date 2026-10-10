#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>   // ★ 关键头文件
#include <vector>
#include <cmath>
#include <ctime>

using namespace cv;
using namespace std;

struct Particle {
    Point2f pos, target;
    float speed;
    Scalar color;
};

int main() {
    srand((unsigned)time(NULL));

    // ★ 1. 创建 FreeType 对象并加载中文字体
    Ptr<freetype::FreeType2> ft2 = freetype::createFreeType2();
    ft2->loadFontData("msyh.ttf", 0); // 请替换为你的字体文件路径

    Mat canvas(600, 600, CV_8UC3, Scalar(0, 0, 0));
    vector<Particle> particles;
    int numParticles = 500;

    for (int i = 0; i < numParticles; i++) {
        Particle p;
        p.pos = Point2f(rand() % 600, rand() % 600);
        double t = (double)rand() / RAND_MAX * 2 * M_PI;
        int x = 300 + 16 * pow(sin(t), 3) * 5;
        int y = 300 - (13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t)) * 5;
        p.target = Point2f(x, y);
        p.speed = 0.05f;
        p.color = Scalar(rand() % 255, rand() % 50, rand() % 255);
        particles.push_back(p);
    }

    while (true) {
        canvas.setTo(Scalar(0, 0, 0));

        for (auto& p : particles) {
            p.pos += (p.target - p.pos) * p.speed;
            circle(canvas, p.pos, 2, p.color, -1);
        }

        // ★ 2. 使用 ft2->putText 绘制中文
        string text = "我爱你石美琼";
        int fontHeight = 40;
        Scalar textColor(0, 0, 255); // 红色
        int thickness = 2;

        // 获取文字尺寸以便居中
        int baseline = 0;
        Size textSize = ft2->getTextSize(text, fontHeight, thickness, &baseline);
        Point textOrg((canvas.cols - textSize.width) / 2,
            (canvas.rows + textSize.height) / 2 - baseline);

        // 先画白色描边（使文字更清晰）
        ft2->putText(canvas, text, textOrg, fontHeight, Scalar(255, 255, 255), thickness + 2,
            LINE_AA, true);
        // 再画红色文字
        ft2->putText(canvas, text, textOrg, fontHeight, textColor, thickness,
            LINE_AA, true);

        imshow("Particle Heart", canvas);
        if (waitKey(30) == 27) break;
    }

    destroyAllWindows();
    return 0;
}