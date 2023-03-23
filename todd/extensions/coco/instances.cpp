#include "instances.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <vector>

namespace py = pybind11;

Category::Category(const py::dict& dict)
    : id(py::cast<Id>(dict["id"])),
      name(py::cast<std::string>(dict["name"])) {}

Image::Image(const py::dict& dict)
    : id(py::cast<Id>(dict["id"])),
      width(py::cast<int>(dict["width"])),
      height(py::cast<int>(dict["height"])),
      filename(py::cast<std::string>(dict["file_name"])) {}

Annotation::Annotation(const py::dict& dict)
    : id(py::cast<Id>(dict["id"])),
      category_id(py::cast<Id>(dict["category_id"])),
      image_id(py::cast<Id>(dict["image_id"])),
      is_crowd(py::cast<int>(dict["iscrowd"])),
      area(py::float_(dict["area"])) {
  py::list bbox = dict["bbox"];
  for (int i = 0; i < bbox.size(); i++) {
    this->bbox[i] = py::float_(bbox[i]);
  }
}

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
  py::module_ json = py::module_::import("json");
  py::dict dict = json.attr("load")(f);
  f.attr("close")();
  return new Annotations(dict);
}
