// #include "predictions.hpp"

// Predictions::Predictions(
//     const std::vector<Category>& categories,
//     const std::vector<Image>& images,
//     const std::vector<Annotation>& annotations,
//     const std::vector<Prediction>& predictions)
//     : Annotations(categories, images, annotations),
//     predictions_(predictions) {
//   std::set<int> image_ids;
//   for (const auto& prediction : predictions) {
//     image_ids.insert(prediction.image_id);
//   }
//   for (const int image_id : image_ids) {
//     if (!this->images().contains(image_id)) {
//       throw std::invalid_argument("Unrecognized image");
//     }
//   }
//   //   Dataset* results = new Dataset(*this);
// }
