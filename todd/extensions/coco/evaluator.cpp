#include "evaluator.hpp"

#include <torch/extension.h>

#include <vector>

#include "instances.hpp"

bool AreaRange::contains(const Object& object) const {
  return min < object.area && object.area < max;
}

Evaluator::Evaluator(const std::vector<AreaRange>& area_ranges)
    : area_ranges_(area_ranges) {}

void Evaluator::evaluate(
    const Annotations& annotations, const Predictions& predictions) const {
  torch::Tensor annotation_bboxes
      = torch::empty({static_cast<int>(annotations.annotations().size()), 4});
  for (const auto& category : annotations.categories()) {
    for (const auto& image : annotations.images()) {
    }
  }
}
