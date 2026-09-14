#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/geometry/fit_rigid.hpp>

namespace {

auto cloud() -> consumer::owned_point_cloud<float, 2> {
  return {trueform_installed_test::points_of<float, 2>(
      {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F})};
}

} // namespace

int main() {
  const auto source = cloud();
  const auto target = cloud();
  const auto result =
      tf::cpp::fit_rigid(source.point_cloud(), target.point_cloud());
  return result.raw_shape() == tf::small_vector<int, 3>{3, 3} ? 0 : 1;
}
