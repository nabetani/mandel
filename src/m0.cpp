#include <boost/program_options.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <vector>

using complex = std::complex<double>;
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
  double t = t0;
  return {colu(t), colu(t + 1), colu(t + 2)};
}

double abslog(double x){
  if ( x==0 ){
    return 0;
  }
  auto l = std::log(std::abs(x)+1);
  auto sign = x<0 ? -1 : 1;
  return l*sign;
}

class mandel_maker {
  cv::Vec3d pix_at(double x, double y) {
    complex z{0.0, 0.0};
    auto c = complex{x, y};
    std::vector<cv::Point2f> points;
    points.reserve(rep);
    for (int ix = 0; ix < rep; ++ix) {
      //z = (std::exp(z) - 1.0 - z*0.6) * 2.0 + c;
      z=z*z+c;
      points.emplace_back(cv::Point2f(abslog(real(z)), abslog(imag(z))));
    }
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);
    auto area = cv::contourArea(hull);
    return {-1, std::arg(z), area};
  }
  int rep;
  int w, h;
  double x0, dx;
  double y0, dy;

public:
  explicit mandel_maker(int rep_, int w_, int h_, double x0_, double x1_,
                        double y0_)
      : rep(rep_), w(w_), h(h_), x0(x0_), dx((x1_ - x0_) / w_), y0(y0_),
        dy((x1_ - x0_) * h_ / w_ / h_) {}
  cv::Mat make() {
    cv::Mat im = cv::Mat::zeros(w, h, CV_64FC3);
#pragma omp parallel for
    for (int iy = 0; iy < h; ++iy) {
      double y = y0 + dy * iy;
      for (int ix = 0; ix < w; ++ix) {
        double x = x0 + ix * dx;
        auto col = pix_at(x, y);
        im.at<cv::Vec3d>(iy, ix) = col;
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
  cv::Mat dest = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
#pragma omp parallel for
  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      auto col{src.at<cv::Vec3d>(y, x)};
      if (col[0] < 0) {
        //auto c0 = color((col[2]*0.1 + col[1] / (PI * 2) * 3 + 3) * 3) / 2;
        auto c0 = color(std::log(col[2]+0.01));
        dest.at<cv::Vec3b>(y, x) = c0;
      } else {
        auto c0 = cv::Vec3b{255, 255, 255} - color(col[1] / (PI * 2) * 3 + 3);
        auto c1 = cv::Vec3b{255, 255, 255} - color(col[0] * 0.01+1);
        constexpr double W = 0.2;
        dest.at<cv::Vec3b>(y, x) = c0 * W + c1 * (1 - W);
      }
    }
  }
  return dest;
}

namespace po = boost::program_options;

po::options_description calc_options() {
  po::options_description desc("calc options");
  desc.add_options()                                                     //
      ("help", "produce help message")                                   //
      ("size,s", po::value<double>()->default_value(60), "size in mm")   //
      ("dpi,d", po::value<double>()->default_value(96), "DPI")           //
      ("rep,r", po::value<int>()->default_value(1000), "repeat count")   //
      ("pos,p", po::value<std::string>()->default_value(""), "position") //
      ;
  return desc;
}

po::options_description col_options() {
  po::options_description desc("col options");
  desc.add_options()                   //
      ("help", "produce help message") //
      ("infile,i", po::value<std::string>()->default_value("data/hoge.yaml"),
       "input yaml file") //
      ("outfile,o", po::value<std::string>()->default_value("data/hoge.png"),
       "output image file") //
      ;
  return desc;
}

int calc(int argc, char const *argv[]) {
  auto desc = calc_options();
  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
    double realsize = vm["size"].as<double>();
    double dpi = vm["dpi"].as<double>();
    auto pix = lround(realsize / 25.4 * dpi);
    omp_set_num_threads(16);
    int rep = vm["rep"].as<int>();
    cv::Rect2d rc = rect(vm["pos"].as<std::string>().c_str());
    auto mm = mandel_maker(rep, pix, pix, rc.x, rc.br().x, rc.y);
    cv::Mat im = mm.make();
    cv::imwrite("data/hoge.png", colorize(im));
    save_image("data/hoge.yaml", im);
    return 0;
  } catch (po::error &e) {
    std::cout << e.what() << std::endl;
    return 0;
  }
}

int colorize(int argc, char const *argv[]) {
  auto desc = col_options();
  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
    auto infile = vm["infile"].as<std::string>();
    auto outfile = vm["outfile"].as<std::string>();
    std::cout << infile << ", " << outfile << "\n";

    cv::FileStorage fs( infile, cv::FileStorage::READ);
    cv::Mat im;
    fs["image"] >> im;
    cv::imwrite(outfile, colorize(im));
    return 0;
  } catch (po::error &e) {
    std::cout << e.what() << std::endl;
    return 0;
  }
}
void show_help() {
  std::cout << calc_options() << "\n" << col_options() << "\n";
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    show_help();
    return 0;
  }
  auto cmd = std::string(argv[1]);
  if (cmd == "calc") {
    return calc(argc - 1, argv + 1);
  } else if (cmd == "col") {
    return colorize(argc - 1, argv + 1);
  } else {
    show_help();
    return 0;
  }
}
