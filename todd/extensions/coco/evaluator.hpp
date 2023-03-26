#pragma once

#include <pybind11/pybind11.h>

#include <array>
#include <vector>

namespace py = pybind11;

class Object;
class Annotations;
class Predictions;

struct AreaRange {
  float min, max;

  bool contains(const Object&) const;
};

class Evaluator {
  std::vector<AreaRange> area_ranges_;

  float iou(const Annotation*, const Prediction*) const;

  void ious(
      std::vector<std::vector<float>>&,
      const std::vector<const Annotation*>&,
      const std::vector<const Prediction*>&) const;

 public:
  Evaluator(const std::vector<AreaRange>&);

  void evaluate(const Annotations&, const Predictions&) const;
};
