#pragma once

#include <pybind11/pybind11.h>

#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace py = pybind11;

using BBox = std::array<float, 4>;

struct Category {
  int id;
  std::string name;

  Category(const py::dict&);
};

struct Image {
  int id;
  int width;
  int height;
  std::string filename;

  Image(const py::dict&);
};

struct Annotation {
  int id;
  int category_id;
  int image_id;
  bool is_crowd;
  BBox bbox;
  float area;

  Annotation(const py::dict&);
};

class Annotations {
  std::map<int, const Category*> categories_;
  std::map<int, const Image*> images_;
  std::map<int, const Annotation*> annotations_;

  std::map<int, std::set<int>> image_to_annotations_;
  std::map<int, std::set<int>> category_to_images_;

 public:
  Annotations(const py::dict&);
  ~Annotations();

  const std::map<int, const Category*>& categories(void) const {
    return categories_;
  }

  const std::map<int, const Image*>& images(void) const { return images_; }

  const std::map<int, const Annotation*>& annotations(void) const {
    return annotations_;
  }

  static Annotations* load(const std::string&);
};
