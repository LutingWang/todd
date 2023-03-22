#include "instances.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <vector>

namespace py = pybind11;

Category::Category(const py::dict& dict)
    : id(py::cast<int>(dict["id"])),
      name(py::cast<std::string>(dict["name"])) {}

Image::Image(const py::dict& dict)
    : id(py::cast<int>(dict["id"])),
      width(py::cast<int>(dict["width"])),
      height(py::cast<int>(dict["height"])),
      filename(py::cast<std::string>(dict["file_name"])) {}

Annotation::Annotation(const py::dict& dict)
    : id(py::cast<int>(dict["id"])),
      category_id(py::cast<int>(dict["category_id"])),
      image_id(py::cast<int>(dict["image_id"])),
      is_crowd(py::cast<bool>(dict["iscrowd"])),
      bbox(py::cast<BBox>(dict["bbox"])),
      area(py::cast<float>(dict["area"])) {}

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
    image_to_annotations_[annotation->image_id].insert(annotation->id);
    category_to_images_[annotation->category_id].insert(annotation->image_id);
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
