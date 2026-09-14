#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/spatial/primitive.hpp>
#include <trueform/cpp/spatial/ray_cast.hpp>

#include <cstdint>

int main() {
  using trueform_installed_test::make_array;
  const consumer::owned_mesh<std::int32_t, float, 3> owned{
      trueform_installed_test::polygons_of<std::int32_t, float, 3>(
          {0, 1, 2},
          {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F})};
  const auto form = owned.mesh();
  const tf::cpp::primitive<double, 3> rays{
      tf::cpp::primitive_kind::ray,
      make_array<double>({0.25, 0.25, -1.0, 0.0, 0.0, 1.0}, {2, 3})};
  const auto result = tf::cpp::ray_cast(rays, form);
  return result.is_scalar() && result.scalar().hit ? 0 : 1;
}
