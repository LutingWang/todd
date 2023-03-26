#include "instances.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <vector>

Category::Category(const py::dict& dict)
    : id(dict["id"]), name(py::cast<std::string>(dict["name"])) {}

Image::Image(const py::dict& dict)
    : id(dict["id"]),
      width(py::cast<int>(dict["width"])),
      height(py::cast<int>(dict["height"])),
      filename(py::cast<std::string>(dict["file_name"])) {}

Object::Object(const py::dict& dict)
    : category_id(py::cast<Id>(dict["category_id"])),
      image_id(py::cast<Id>(dict["image_id"])) {
  py::list bbox = dict["bbox"];
  for (int i = 0; i < bbox.size(); i++) {
    this->bbox[i] = py::float_(bbox[i]);
  }
  py::float_ area = dict.contains("area") ? dict["area"] : bbox[2] * bbox[3];
  this->area = area;
}

Annotation::Annotation(const py::dict& dict)
    : Object(dict), id(dict["id"]), is_crowd(py::cast<int>(dict["iscrowd"])) {}

Annotations::Annotations(const py::dict& dict) {
  for (const auto& category_dict : dict["categories"]) {
    const Category* category = new Category(py::cast<py::dict>(category_dict));
    categories_[category->id] = category;
  }
  for (const auto& image_dict : dict["images"]) {
    const Image* image = new Image(py::cast<py::dict>(image_dict));
    images_[image->id] = image;
  }
  for (const auto& annotation_dict : dict["annotations"]) {
    const Annotation* annotation
        = new Annotation(py::cast<py::dict>(annotation_dict));
    annotations_[annotation->id] = annotation;
    category_id_to_image_id_to_annotation_[annotation->category_id]
                                          [annotation->image_id]
                                              .push_back(annotation);
  }
}

Annotations::~Annotations() {
  for (const auto& category : categories_) {
    delete category.second;
  }
  for (const auto& image : images_) {
    delete image.second;
  }
  for (const auto& annotation : annotations_) {
    delete annotation.second;
  }
}

Annotations* Annotations::load(const std::string& filename) {
  py::module_ builtins = py::module_::import("builtins");
  py::object f = builtins.attr("open")(filename);
  try {
    py::module_ json = py::module_::import("json");
    py::dict dict = json.attr("load")(f);
    return new Annotations(dict);
  } catch (...) {
    f.attr("close")();
    throw;
  }
}

Prediction::Prediction(const py::dict& dict)
    : Object(dict), score(py::cast<float>(dict["score"])) {}

Predictions::Predictions(const py::list& list) {
  for (const auto& prediction_dict : list) {
    const Prediction* prediction
        = new Prediction(py::cast<py::dict>(prediction_dict));
    predictions_.push_back(prediction);
    category_id_to_image_id_to_prediction_[prediction->category_id]
                                          [prediction->image_id]
                                              .push_back(prediction);
  }
}

Predictions::~Predictions(void) {
  for (const Prediction* prediction : predictions_) {
    delete prediction;
  }
}

Predictions* Predictions::load(const std::string& filename) {
  py::module_ builtins = py::module_::import("builtins");
  py::object f = builtins.attr("open")(filename);
  try {
    py::module_ json = py::module_::import("json");
    py::list list = json.attr("load")(f);
    return new Predictions(list);
  } catch (...) {
    f.attr("close")();
    throw;
  }
}
