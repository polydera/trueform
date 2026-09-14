#include <trueform/cpp/core/nd_array.hpp>
#include <trueform/cpp/spatial/async/closest_metric_point_pair.hpp>
#include <trueform/cpp/spatial/closest_metric_point_pair.hpp>
#include <trueform/cpp/spatial/primitive.hpp>

#include <future>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <variant>

namespace {

template <typename Real>
auto make_array(std::initializer_list<Real> values,
                tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<Real> {
  tf::buffer<Real> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<Real>::from_buffer(std::move(buffer),
                                              std::move(shape));
}

auto point() -> tf::cpp::primitive<float, 2> {
  return tf::cpp::primitive<float, 2>(tf::cpp::primitive_kind::point,
                                      make_array<float>({0.0F, 2.0F}, {2}));
}

auto segment() -> tf::cpp::primitive<float, 2> {
  return tf::cpp::primitive<float, 2>(
      tf::cpp::primitive_kind::segment,
      make_array<float>({1.0F, 0.0F, 3.0F, 0.0F}, {2, 2}));
}

using result_type =
    std::variant<tf::cpp::closest_metric_point_pair_result<float>,
                 tf::cpp::closest_metric_point_pair_batch_result<float>>;

auto is_exact_pair(const result_type &value) -> bool {
  const auto *result =
      std::get_if<tf::cpp::closest_metric_point_pair_result<float>>(&value);
  return result && result->point0.ndim() == 1 &&
         result->point0.shape_at(0) == 2 && result->point0[0] == 0.0F &&
         result->point0[1] == 2.0F && result->point1.ndim() == 1 &&
         result->point1.shape_at(0) == 2 && result->point1[0] == 1.0F &&
         result->point1[1] == 0.0F && result->distance2 == 5.0F;
}

} // namespace

int main() {
  const auto first = point();
  const auto second = segment();
  const auto sync_result = tf::cpp::closest_metric_point_pair(first, second);
  auto pending_result =
      tf::cpp::async::closest_metric_point_pair(first, second);
  static_assert(std::is_same_v<decltype(sync_result), const result_type>);
  static_assert(
      std::is_same_v<decltype(pending_result), std::future<result_type>>);
  return is_exact_pair(sync_result) && is_exact_pair(pending_result.get()) ? 0
                                                                           : 1;
}
