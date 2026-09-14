#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/geometry/area.hpp>
#include <trueform/cpp/geometry/mean_edge_length.hpp>

#include <cstdint>

int main() {
  using real_type = float;
  using index_type = std::int64_t;
  constexpr std::size_t dims = 2;
  const consumer::owned_mesh<index_type, real_type, dims> value{
      trueform_installed_test::polygons_of<index_type, real_type, dims>(
          {0, 1, 2}, {0, 0, 1, 0, 0, 1})};

  const auto area = tf::cpp::area<index_type, real_type, dims>(value.mesh());
  const auto mean =
      tf::cpp::mean_edge_length<index_type, real_type, dims>(value.mesh());
  return area == real_type{0.5} && mean > real_type{1} && mean < real_type{1.2}
             ? 0
             : 1;
}
