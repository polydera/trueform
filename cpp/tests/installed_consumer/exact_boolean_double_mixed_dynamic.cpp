#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/csg/make_boolean.hpp>

#include <cstdint>
#include <type_traits>

int main() {
  using namespace trueform_installed_test;
  using real_type = double;
  using output_index = std::int64_t;

  const auto dynamic_narrow =
      empty_mesh_3d<std::int32_t, real_type, tf::dynamic_size>();
  const auto fixed_wide = empty_mesh_3d<output_index, real_type>();
  const auto result =
      tf::cpp::make_boolean<std::int32_t, real_type, output_index>(
          dynamic_narrow.mesh(), fixed_wide.mesh(), tf::boolean_op::merge);
  const auto with_curves =
      tf::cpp::make_boolean_with_curves<std::int32_t, real_type, output_index>(
          dynamic_narrow.mesh(), fixed_wide.mesh(), tf::boolean_op::merge);

  static_assert(
      std::is_same_v<
          std::decay_t<decltype(result)>,
          tf::cpp::boolean_result<output_index, real_type, tf::dynamic_size>>);
  static_assert(std::is_same_v<std::decay_t<decltype(with_curves)>,
                               tf::cpp::boolean_with_curves_result<
                                   output_index, real_type, tf::dynamic_size>>);
  return result.mesh.size() == 0 && with_curves.mesh.size() == 0 &&
                 with_curves.curves.size() == 0
             ? 0
             : 1;
}
