#include <cmath>
#include <complex>
#include <cstdint>
#include <omp.h>
#include <opencv2/opencv.hpp>

using complex = std::complex<double>;
constexpr int N = 10000;
constexpr double PI = 3.14159'26535'89793'23846'26433'83279'50288;

std::uint8_t colu(double t0) {
  double t = std::fmod(t0, 3.0);
  if (t < 2) {
    return static_cast<std::uint8_t>((1 - std::cos(t * PI)) * (255 / 2.0));
  } else {
    return 0;
  }
}

cv::Vec3b color(int i) {
  double t = std::log(1.0 + i);
  return {colu(t), colu(t + 1), colu(t + 2)};
}

class mandel_maker {

  cv::Vec3b pix_at(double x, double y) {
    complex z{0.0, 0.0};
    complex c{x, y};
    for (int i = 0; i < N; ++i) {
      z = std::pow(z, std::abs(z) / 2 + 2) + c;
      if (10.0 < std::abs(z)) {
        return color(i);
      }
    }
    return {0, 0, 0};
  }
  int w, h;
  double x0, dx;
  double y0, dy;

public:
  explicit mandel_maker(int w_, int h_, double x0_, double x1_, double y0_)
      : w(w_), h(h_), x0(x0_), dx((x1_ - x0_) / w_), y0(y0_),
        dy((x1_ - x0_) * h_ / w_ / h_) {}
  cv::Mat make() {
    cv::Mat im = cv::Mat::zeros(w, h, CV_8UC3);
#pragma omp parallel for
    for (int iy = 0; iy < h; ++iy) {
      double y = y0 + dy * iy;
      for (int ix = 0; ix < w; ++ix) {
        double x = x0 + ix * dx;
        auto col = pix_at(x, y);
        im.at<cv::Vec3b>(iy, ix) = col;
      }
    }
    return im;
  }
};

cv::Rect2d rect(char const *cmd) {
  cv::Rect2d r(-1, -1, 2, 2);
  for (; *cmd; ++cmd) {
    double x, y;
    double w = r.width / 2;
    double h = r.height / 2;
    int c = *cmd - '0';
    switch (c % 3) {
    case 0:
      x = r.x;
      break;
    case 1:
      x = r.x + w / 2;
      break;
    case 2:
      x = r.x + w;
      break;
    }
    switch (c / 3) {
    case 0:
      y = r.y;
      break;
    case 1:
      y = r.y + h / 2;
      break;
    case 2:
      y = r.y + h;
      break;
    }
    r = cv::Rect2d(x, y, w, h);
  }
  return r;
}

int main(int argc, char const *argv[]) {
  double realsize = argc < 2 ? 100.0 : std::atof(argv[1]); // size in mm
  omp_set_num_threads(16);
  int pix = static_cast<int>(realsize / 25.4 * 200); // 200dpi
  cv::Rect2d rc = rect(argc < 3 ? "" : argv[2]);
  auto mm = mandel_maker(pix, pix, rc.x, rc.br().x, rc.y);
  cv::Mat im = mm.make();
  cv::imwrite("data/hoge.png", im);
}
