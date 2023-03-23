#pragma once

#include <pybind11/pybind11.h>

#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace py = pybind11;

using Id = py::int_;
using BBox = std::array<float, 4>;

struct Category {
  Id id;
  std::string name;

  Category(const py::dict&);
};

struct Image {
  Id id;
  int width;
  int height;
  std::string filename;

  Image(const py::dict&);
};

struct Annotation {
  Id id;
  Id category_id;
  Id image_id;
  bool is_crowd;
  BBox bbox;
  float area;

  Annotation(const py::dict&);
};

class Annotations {
  std::map<Id, const Category*> categories_;
  std::map<Id, const Image*> images_;
  std::map<Id, const Annotation*> annotations_;

 public:
  Annotations(const py::dict&);
  ~Annotations();

  const std::map<Id, const Category*>& categories(void) const {
    return categories_;
  }

  const std::map<Id, const Image*>& images(void) const { return images_; }

  const std::map<Id, const Annotation*>& annotations(void) const {
    return annotations_;
  }

  static Annotations* load(const std::string&);
};
