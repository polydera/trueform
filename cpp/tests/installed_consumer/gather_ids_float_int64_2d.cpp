#include "spatial_float_int64_2d_test_utils.hpp"

#include <trueform/cpp/spatial/gather_ids.hpp>
#include <trueform/cpp/spatial/gather_ids_within_distance.hpp>

#include <cstdint>
#include <type_traits>

int main() {
  const auto owned = spatial_float_int64_2d_test::triangle_mesh();
  const auto form = owned.mesh();
  const auto query = spatial_float_int64_2d_test::point();
  const auto exact = tf::cpp::gather_ids(form, query);
  const auto nearby = tf::cpp::gather_ids_within_distance(form, query, 0.0F);
  static_assert(
      std::is_same_v<decltype(exact), const tf::cpp::nd_array<std::int64_t>>);
  return exact.length() == 1 && exact[0] == 0 && nearby.length() == 1 &&
                 nearby[0] == 0
             ? 0
             : 1;
}
