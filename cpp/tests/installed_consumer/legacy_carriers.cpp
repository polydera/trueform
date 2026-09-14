#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/core/build_tree.hpp>
#include <trueform/cpp/spatial/async/distance.hpp>
#include <trueform/cpp/spatial/distance.hpp>
#include <trueform/cpp/spatial/primitive.hpp>

template <typename Real> auto check_legacy_carriers() -> bool {
  using namespace trueform_installed_test;
  const consumer::owned_mesh<tf::cpp::default_index_t, Real, 3> mesh{
      polygons_of<tf::cpp::default_index_t, Real, 3>(
          {0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0})};
  const auto assembled = mesh.mesh();
  tf::cpp::build_tree(assembled);

  const consumer::owned_point_cloud<Real, 3> cloud{
      points_of<Real, 3>({0, 0, 0, 1, 2, 3})};
  const auto assembled_cloud = cloud.point_cloud();
  tf::cpp::build_tree(assembled_cloud);

  auto primitive =
      tf::cpp::primitive<Real>(tf::cpp::primitive_kind::point,
                               make_array<Real>({0, 0, 0, 1, 2, 3}, {2, 3}));
  const auto first = primitive.at(0);
  const auto second = primitive.at(1);
  const auto sync_distance = tf::cpp::distance(first, second).scalar();
  const auto async_distance2 =
      tf::cpp::async::distance2(first, second).get().scalar();

  return mesh.cache.is_tree_fresh(assembled.geometry()) &&
         cloud.cache.is_tree_fresh(assembled_cloud.geometry()) &&
         primitive.count() == 2 && sync_distance > Real{0} &&
         async_distance2 > Real{0};
}

int main() {
  return check_legacy_carriers<float>() && check_legacy_carriers<double>() ? 0
                                                                          : 1;
}
