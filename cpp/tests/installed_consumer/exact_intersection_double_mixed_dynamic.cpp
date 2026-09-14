#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/intersect/intersection_curves.hpp>
#include <trueform/cpp/intersect/self_intersection_curves.hpp>

#include <cstdint>
#include <type_traits>
#include <vector>

int main() {
  using namespace trueform_installed_test;
  using real_type = double;
  using wide_index = std::int64_t;
  using curves_type = tf::curves_buffer<wide_index, real_type, 3>;

  const auto dynamic_narrow =
      empty_mesh_3d<std::int32_t, real_type, tf::dynamic_size>();
  const auto fixed_wide = empty_mesh_3d<wide_index, real_type>();
  const auto mixed =
      tf::cpp::intersection_curves(dynamic_narrow.mesh(), fixed_wide.mesh());
  static_assert(std::is_same_v<std::decay_t<decltype(mixed)>, curves_type>);

  // a range is homogeneous, so its element states the arity
  const auto first = empty_mesh_3d<wide_index, real_type, tf::dynamic_size>();
  const auto second = empty_mesh_3d<wide_index, real_type, tf::dynamic_size>();
  const std::vector<tf::cpp::mesh<wide_index, real_type, 3, tf::dynamic_size>>
      forms{first.mesh(), second.mesh()};
  const auto multiple = tf::cpp::intersection_curves(forms);
  const auto self = tf::cpp::self_intersection_curves(first.mesh());
  static_assert(std::is_same_v<std::decay_t<decltype(multiple)>, curves_type>);
  static_assert(std::is_same_v<std::decay_t<decltype(self)>, curves_type>);

  return mixed.size() == 0 && multiple.size() == 0 && self.size() == 0 ? 0 : 1;
}
