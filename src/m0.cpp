#include <complex>
#include <cmath>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <omp.h>

using complex = std::complex<double>;
constexpr int N = 3000;
constexpr double PI = 3.14159'26535'89793'23846'26433'83279'50288;

std::uint8_t colu(double t0)
{
  double t = std::fmod( t0, 3.0 );
  if ( t<2 ){
    return static_cast<std::uint8_t>((1-std::cos(t*PI)) * (255/2.0));
  } else {
    return 0;
  }
}

cv::Vec3b color( int i )
{
  double t = std::log(1.0+i);
  return { colu(t), colu(t+1), colu(t+2) };
}


class mandel_maker {

  cv::Vec3b pix_at(double x, double y) {
    complex z{0.0, 0.0};
    complex c{x, y};
    for (int i = 0; i < N; ++i) {
      z = z * z + c;
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
      : w(w_), h(h_), x0(x0_), dx((x1_ - x0_)/w_), y0(y0_),
        dy((x1_ - x0_) * h_ / w_ / h_) {}
  cv::Mat make()
  {
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

int main( int argc, char const * argv[] ) {
  double realsize = argc<2 ? 100.0 : std::atof(argv[1]); // size in mm
  int pix = static_cast<int>( realsize / 25.4 * 200 ); // 200dpi
  auto mm = mandel_maker(pix, pix, -2, 0.5, -1.25 );
  cv::Mat im = mm.make();
  cv::imwrite("data/hoge.png", im);
}
