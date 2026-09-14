#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/geometry/positively_oriented.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

template <typename Index, typename Real>
auto signed_volume6(const tf::polygons_buffer<Index, Real, 3, 3> &value)
    -> Real {
  auto result = Real{0};
  for (const auto face : value.polygons()) {
    const auto &p0 = face[0];
    const auto &p1 = face[1];
    const auto &p2 = face[2];
    result += p0[0] * (p1[1] * p2[2] - p1[2] * p2[1]) +
              p0[1] * (p1[2] * p2[0] - p1[0] * p2[2]) +
              p0[2] * (p1[0] * p2[1] - p1[1] * p2[0]);
  }
  return result;
}

} // namespace

int main() {
  using real_type = double;
  using index_type = std::int64_t;
  using storage_type = tf::polygons_buffer<index_type, real_type, 3, 3>;
  const auto value =
      trueform_installed_test::negative_tetrahedron<index_type, real_type>();
  const auto result = tf::cpp::positively_oriented<index_type, real_type, 3>(
      value.mesh(), true);
  static_assert(std::is_same_v<std::decay_t<decltype(result)>, storage_type>);
  return result.size() == 4 && result.points_buffer().size() == 4 &&
                 signed_volume6(result) > 0
             ? 0
             : 1;
}
