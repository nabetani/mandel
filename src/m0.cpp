#include <complex>
#include <opencv2/opencv.hpp>

#include <omp.h>

using complex = std::complex<double>;
constexpr int N = 1000;

cv::Vec3b pix_at(double x, double y) {
  complex z{0.0, 0.0};
  complex c{x, y};
  for (int i = 0; i < N; ++i) {
    z = z * z + c;
    if (10.0 < std::abs(z)) {
      return {0, 0, 0};
    }
  }
  return {200, 200, 200};
}

cv::Mat mandel(int w, int h) {
  cv::Mat im = cv::Mat::zeros(w, h, CV_8UC3);
#pragma omp parallel for
  for (int iy = 0; iy < h; ++iy) {
    double y = (iy - h / 2) / (h / 2.0)*2;
    for (int ix = 0; ix < w; ++ix) {
      double x = (ix - w / 2) / (w / 2.0)*2;
      auto col = pix_at(x, y);
      im.at<cv::Vec3b>(ix, iy) = col;
    }
  }
  return im;
}

int main() {

  cv::Mat im = mandel(800, 800);
  cv::imwrite("data/hoge.png", im);
}
