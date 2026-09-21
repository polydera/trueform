#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/spatial/distance.hpp>
#include <trueform/cpp/spatial/primitive.hpp>

namespace {

auto point(float x, float y, float z) -> tf::cpp::primitive<float, 3> {
  return tf::cpp::primitive<float, 3>(
      tf::cpp::primitive_kind::point,
      trueform_installed_test::make_array<float>({x, y, z}, {3}));
}

auto cloud() -> consumer::owned_point_cloud<float, 3> {
  return {trueform_installed_test::points_of<float, 3>(
      {0.0F, 0.0F, 0.0F, 3.0F, 0.0F, 0.0F})};
}

} // namespace

int main() {
  const auto query = point(3.0F, 4.0F, 0.0F);
  const auto between_primitives =
      tf::cpp::distance(point(0.0F, 0.0F, 0.0F), query);
  const auto owned = cloud();
  const auto from_cloud = tf::cpp::distance(owned.point_cloud(), query);
  return between_primitives.is_scalar() &&
                 between_primitives.scalar() == 5.0F &&
                 from_cloud.is_scalar() && from_cloud.scalar() == 4.0F
             ? 0
             : 1;
}
