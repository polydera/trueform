#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/iso/isocontours.hpp>

#include <cstdint>
#include <type_traits>

int main() {
  using namespace trueform_installed_test;
  using real_type = double;
  using index_type = std::int64_t;
  using curves_type = tf::curves_buffer<index_type, real_type, 3>;

  const auto scalars = make_array<real_type>({}, {0});
  const auto cuts = make_array<real_type>({0, 1}, {2});
  const auto owned = empty_mesh_3d<index_type, real_type, tf::dynamic_size>();
  const auto single = tf::cpp::isocontours<index_type, real_type>(
      owned.mesh(), scalars, real_type{0});
  const auto multiple =
      tf::cpp::isocontours<index_type, real_type>(owned.mesh(), scalars, cuts);

  static_assert(std::is_same_v<std::decay_t<decltype(single)>, curves_type>);
  static_assert(std::is_same_v<std::decay_t<decltype(multiple)>, curves_type>);
  return single.size() == 0 && multiple.size() == 0 ? 0 : 1;
}
