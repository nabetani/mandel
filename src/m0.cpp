#include <cmath>
#include <complex>
#include <cstdint>
#include <omp.h>
#include <opencv2/opencv.hpp>

using complex = std::complex<double>;
constexpr int N = 100000;
constexpr double PI = 3.14159'26535'89793'23846'26433'83279'50288;

std::uint8_t colu(double t0) {
  double t = std::fmod(t0, 3.0);
  if (t < 2) {
    return static_cast<std::uint8_t>(
        std::lround((1 - std::cos(t * PI)) * (255 / 2.0)));
  } else {
    return 0;
  }
}

cv::Vec3b color(double t0) {
  double t = std::log(t0 + 1);
  return {colu(t), colu(t + 1), colu(t + 2)};
}

class mandel_maker {

  int32_t pix_at(double x, double y) {
    complex z{0.0, 0.0};
    auto c = complex{x, y};
    for (int ix = 0; ix < N; ++ix) {
      z = z * z + c;
      if (2.0 < std::abs(z)) {
        return ix;
      }
    }
    return -1;
  }
  int w, h;
  double x0, dx;
  double y0, dy;

public:
  explicit mandel_maker(int w_, int h_, double x0_, double x1_, double y0_)
      : w(w_), h(h_), x0(x0_), dx((x1_ - x0_) / w_), y0(y0_),
        dy((x1_ - x0_) * h_ / w_ / h_) {}
  cv::Mat make() {
    cv::Mat im = cv::Mat::zeros(w, h, CV_32S);
#pragma omp parallel for
    for (int iy = 0; iy < h; ++iy) {
      double y = y0 + dy * iy;
      for (int ix = 0; ix < w; ++ix) {
        double x = x0 + ix * dx;
        auto col = pix_at(x, y);
        im.at<std::int32_t>(iy, ix) = col;
      }
    }
    return im;
  }
};

cv::Rect2d rect(char const *cmd) {
  double w0 = 1 << 5;
  cv::Rect2d r(-w0 / 2, -w0 / 2, w0, w0);
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

void save_image(char const *filename, cv::Mat const &im) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs << "image" << im;
}

std::pair<std::int32_t, std::int32_t> find_minmax(cv::Mat const &im) {
  int min = INT32_MAX;
  int max = INT32_MIN;
  auto end = im.end<std::int32_t>();
  for (auto it = im.begin<std::int32_t>(); it != end; ++it) {
    int32_t col = *it;
    if (col < 0) {
      continue;
    }
    if (col < min) {
      min = col;
    }
    if (max < col) {
      max = col;
    }
  }
  return {min, max};
}

cv::Mat colorize(cv::Mat const &src) {
  auto [min, max] = find_minmax(src);
  cv::Mat dest = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
  double colfactor = 255.0 / (max - min);
  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      auto col{src.at<std::int32_t>(y, x)};
      if (col < 0) {
        dest.at<cv::Vec3b>(y, x) = {0, 0, 0};
      } else {
        dest.at<cv::Vec3b>(y, x) = color(col);
      }
    }
  }
  return dest;
}

int main(int argc, char const *argv[]) {
  double realsize = argc < 2 ? 100.0 : std::atof(argv[1]); // size in mm
  omp_set_num_threads(16);
  int pix = static_cast<int>(realsize / 25.4 * 200); // 200dpi
  cv::Rect2d rc = rect(argc < 3 ? "" : argv[2]);
  auto mm = mandel_maker(pix, pix, rc.x, rc.br().x, rc.y);
  cv::Mat im = mm.make();
  cv::imwrite("data/hoge.png", colorize(im));
  save_image("data/hoge.yaml", im);
}
