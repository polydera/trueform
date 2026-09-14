#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/iso/isobands.hpp>

#include <cstdint>
#include <type_traits>

int main() {
  using namespace trueform_installed_test;
  using real_type = double;
  using index_type = std::int64_t;

  const auto scalars = make_array<real_type>({}, {0});
  const auto cuts = make_array<real_type>({0}, {1});
  const auto selected = make_array<std::int32_t>({0}, {1});

  const auto owned = empty_mesh_3d<index_type, real_type, tf::dynamic_size>();
  const auto result =
      tf::cpp::isobands<index_type, real_type, tf::dynamic_size>(owned.mesh(),
                                                                 scalars, cuts);
  const auto with_curves =
      tf::cpp::isobands_with_curves<index_type, real_type, tf::dynamic_size>(
          owned.mesh(), scalars, cuts);
  const auto selected_result =
      tf::cpp::isobands_selected<index_type, real_type, tf::dynamic_size>(
          owned.mesh(), scalars, cuts, selected);
  const auto selected_with_curves =
      tf::cpp::isobands_with_curves_selected<index_type, real_type,
                                             tf::dynamic_size>(
          owned.mesh(), scalars, cuts, selected);

  static_assert(
      std::is_same_v<
          std::decay_t<decltype(result)>,
          tf::cpp::isobands_result<index_type, real_type, tf::dynamic_size>>);
  static_assert(
      std::is_same_v<std::decay_t<decltype(with_curves)>,
                     tf::cpp::isobands_with_curves_result<index_type, real_type,
                                                          tf::dynamic_size>>);
  return result.mesh.size() == 0 && with_curves.curves.size() == 0 &&
                 selected_result.mesh.size() == 0 &&
                 selected_with_curves.curves.size() == 0
             ? 0
             : 1;
}
