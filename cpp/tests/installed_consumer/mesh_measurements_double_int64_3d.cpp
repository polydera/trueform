#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/geometry/area.hpp>
#include <trueform/cpp/geometry/mean_edge_length.hpp>
#include <trueform/cpp/geometry/signed_volume.hpp>
#include <trueform/cpp/geometry/volume.hpp>

#include <cmath>
#include <cstdint>

int main() {
  using real_type = double;
  using index_type = std::int64_t;
  constexpr std::size_t dims = 3;
  const consumer::owned_mesh<index_type, real_type, dims> value{
      trueform_installed_test::polygons_of<index_type, real_type, dims>(
          {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3},
          {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1})};

  const auto area = tf::cpp::area<index_type, real_type, dims>(value.mesh());
  const auto mean =
      tf::cpp::mean_edge_length<index_type, real_type, dims>(value.mesh());
  const auto signed_volume =
      tf::cpp::signed_volume<index_type, real_type, dims>(value.mesh());
  const auto volume =
      tf::cpp::volume<index_type, real_type, dims>(value.mesh());
  return area > real_type{2} && mean > real_type{1} &&
                 std::abs(signed_volume) == volume &&
                 volume == real_type{1} / real_type{6}
             ? 0
             : 1;
}
