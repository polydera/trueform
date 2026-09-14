#include "spatial_float_int64_2d_test_utils.hpp"

#include <trueform/cpp/spatial/neighbor_search.hpp>

#include <cstdint>
#include <type_traits>

int main() {
  const auto owned = spatial_float_int64_2d_test::triangle_mesh();
  const auto form = owned.mesh();
  const auto query = spatial_float_int64_2d_test::point();
  const auto result = tf::cpp::neighbor_search(form, query);
  static_assert(
      std::is_same_v<decltype(result),
                     const tf::cpp::neighbor_result<std::int64_t, float, 2>>);
  return result.element_id == 0 && result.distance2 == 0.0F &&
                 result.point.raw_shape() == tf::small_vector<int, 3>{2}
             ? 0
             : 1;
}
