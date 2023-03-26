#pragma once

#include <pybind11/pybind11.h>

#include <array>
#include <cassert>
#include <map>
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

struct Object {
  Id category_id;
  Id image_id;
  BBox bbox;
  float area;

  Object(const py::dict&);
};

struct Annotation : Object {
  Id id;
  bool is_crowd;

  Annotation(const py::dict&);
};

class Annotations final {
  std::map<Id, const Category*> categories_;
  std::map<Id, const Image*> images_;
  std::map<Id, const Annotation*> annotations_;
  std::map<Id, std::map<Id, std::vector<const Annotation*>>>
      category_id_to_image_id_to_annotation_;

 public:
  Annotations(const py::dict&);
  ~Annotations(void);

  const std::map<Id, const Category*>& categories(void) const {
    return categories_;
  }

  const std::map<Id, const Image*>& images(void) const { return images_; }

  const std::map<Id, const Annotation*>& annotations(void) const {
    return annotations_;
  }

  const std::vector<const Annotation*>& annotations(
      Id category_id, Id image_id) {
    assert(categories_.contains(category_id));
    assert(images_.contains(image_id));
    return category_id_to_image_id_to_annotation_[category_id][image_id];
  }

  static Annotations* load(const std::string&);
};

struct Prediction : public Object {
  float score;

  Prediction(const py::dict&);
};

class Predictions final {
  std::vector<const Prediction*> predictions_;
  std::map<Id, std::map<Id, std::vector<const Prediction*>>>
      category_id_to_image_id_to_prediction_;

 public:
  Predictions(const py::list&);
  ~Predictions(void);

  const std::vector<const Prediction*>& predictions(void) const {
    return predictions_;
  }

  static Predictions* load(const std::string&);
};
