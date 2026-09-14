#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/geometry/principal_curvatures.hpp>
#include <trueform/cpp/geometry/principal_directions.hpp>
#include <trueform/cpp/geometry/shape_index.hpp>

#include <cstdint>
#include <type_traits>

int main() {
  using real_type = double;
  using index_type = std::int64_t;
  const consumer::owned_mesh<index_type, real_type, 3, tf::dynamic_size> value{
      trueform_installed_test::polygons_of<index_type, real_type, 3>(
          {0, 3, 7}, {0, 1, 2, 1, 3, 4, 2},
          {0, 0, 0, 1, 0, 0, 0.5, 1, 0, 2, 0, 0, 1.5, 1, 0})};

  const auto curvatures =
      tf::cpp::principal_curvatures<index_type, real_type, 3>(value.mesh(), 2);
  const auto directions =
      tf::cpp::principal_directions<index_type, real_type, 3>(value.mesh(), 2);
  const auto shape =
      tf::cpp::shape_index<index_type, real_type, 3>(value.mesh(), 2);
  static_assert(
      std::is_same_v<decltype(curvatures.k0), tf::cpp::nd_array<real_type>>);
  return curvatures.k0.length() == 5 && curvatures.k1.length() == 5 &&
                 directions.d0.length() == 15 &&
                 directions.d1.length() == 15 && shape.length() == 5 &&
                 curvatures.k0[0] == real_type{0} &&
                 curvatures.k1[4] == real_type{0} && shape[2] == real_type{0}
             ? 0
             : 1;
}
