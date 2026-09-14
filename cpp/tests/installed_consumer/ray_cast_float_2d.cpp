#include "spatial_float_int64_2d_test_utils.hpp"

#include <trueform/cpp/spatial/ray_cast.hpp>

#include <cstdint>
#include <type_traits>

int main() {
  const auto owned = spatial_float_int64_2d_test::triangle_mesh();
  const auto form = owned.mesh();
  const auto ray = spatial_float_int64_2d_test::ray();
  const auto result = tf::cpp::ray_cast(ray, form);
  static_assert(
      std::is_same_v<decltype(result),
                     const tf::cpp::ray_cast_form_result<std::int64_t, float>>);
  return result.is_scalar() && result.scalar().hit &&
                 result.scalar().t == 1.0F && result.scalar().element_id == 0
             ? 0
             : 1;
}
