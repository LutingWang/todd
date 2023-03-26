#include <torch/extension.h>

#include <array>
#include <string>
#include <vector>

#include "instances.hpp"

constexpr int kMaxOther = 500;

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
  py::class_<Object>(m, "Object")
      .def(py::init<const Id&, const Id&, const py::list&>())
      .def_readwrite("category_id", &Object::category_id)
      .def_readwrite("image_id", &Object::image_id)
      .def_readwrite("bbox", &Object::bbox)
      .def("__repr__", [](const Object& o) {
        return "Object_({}, {}, {})"_s.format(
            o.category_id, o.image_id, o.bbox);
      });
  py::class_<Annotation, Object>(m, "Annotation")
      .def(py::init<const py::dict&>())
      .def_readwrite("id_", &Annotation::id)
      .def_readwrite("is_crowd", &Annotation::is_crowd)
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
      .def_static("load", &Annotations::load)
      .def("__repr__", [](const Annotations& a) {
        std::vector<const Category*> categories;
        std::vector<const Image*> images;
        std::vector<const Annotation*> annotations;
        for (const auto& category : a.categories()) {
          categories.push_back(category.second);
        }
        for (const auto& image : a.images()) {
          images.push_back(image.second);
        }
        for (const auto& annotation : a.annotations()) {
          annotations.push_back(annotation.second);
        }
        py::object Repr = py::module_::import("reprlib").attr("Repr");
        py::object repr = Repr();
        repr.attr("maxother") = kMaxOther;
        return "Annotations({})"_s.format(py::dict(
            "categories"_a = repr.attr("repr")(categories),
            "images"_a = repr.attr("repr")(images),
            "annotations"_a = repr.attr("repr")(annotations)));
      });
  py::class_<Prediction, Object>(m, "Prediction")
      .def(py::init<const py::dict&>())
      .def_readwrite("score", &Prediction::score)
      .def("__repr__", [](const Prediction& p) {
        return "Prediction({})"_s.format(py::dict(
            "category_id"_a = p.category_id,
            "image_id"_a = p.image_id,
            "bbox"_a = p.bbox,
            "score"_a = p.score));
      });
  py::class_<Predictions>(m, "Predictions")
      .def(py::init<const py::list&>())
      .def_property_readonly("predictions", &Predictions::predictions)
      .def_static("load", &Predictions::load)
      .def("__repr__", [](const Predictions& p) {
        py::object Repr = py::module_::import("reprlib").attr("Repr");
        py::object repr = Repr();
        repr.attr("maxother") = kMaxOther;
        return "Predictions({})"_s.format(repr.attr("repr")(p.predictions()));
      });
}
