#include <torch/extension.h>

#include <array>
#include <string>
#include <vector>

#include "instances.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace py::literals;
  py::class_<Category>(m, "Category")
      .def(py::init<const py::dict&>())
      .def_readwrite("id_", &Category::id)
      .def_readwrite("name", &Category::name)
      .def("__repr__", [](const Category& c) {
        return "Category({})"_s.format(
            py::dict("id"_a = c.id, "name"_a = c.name));
      });
  py::class_<Image>(m, "Image")
      .def(py::init<const py::dict&>())
      .def_readwrite("id_", &Image::id)
      .def_readwrite("width", &Image::width)
      .def_readwrite("height", &Image::height)
      .def_readwrite("filename", &Image::filename)
      .def("__repr__", [](const Image& i) {
        return "Image({})"_s.format(py::dict(
            "id"_a = i.id,
            "width"_a = i.width,
            "height"_a = i.height,
            "file_name"_a = i.filename));
      });
  py::class_<Annotation>(m, "Annotation")
      .def(py::init<const py::dict&>())
      .def_readwrite("id_", &Annotation::id)
      .def_readwrite("category_id", &Annotation::category_id)
      .def_readwrite("image_id", &Annotation::image_id)
      .def_readwrite("is_crowd", &Annotation::is_crowd)
      //   .def_readwrite("bbox", &Annotation::bbox)
      .def_readwrite("area", &Annotation::area)
      .def("__repr__", [](const Annotation& a) {
        return "Annotation({})"_s.format(py::dict(
            "id"_a = a.id,
            "category_id"_a = a.category_id,
            "image_id"_a = a.image_id,
            "iscrowd"_a = a.is_crowd,
            "bbox"_a = a.bbox,
            "area"_a = a.area));
      });
  py::class_<Annotations>(m, "Annotations")
      .def(py::init<const py::dict&>())
      .def_property_readonly("categories", &Annotations::categories)
      .def_property_readonly("images", &Annotations::images)
      .def_property_readonly("annotations", &Annotations::annotations)
      .def_static("load", &Annotations::load);
}
