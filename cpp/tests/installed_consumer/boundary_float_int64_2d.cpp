#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/topology/boundary_curves.hpp>
#include <trueform/cpp/topology/boundary_edges.hpp>
#include <trueform/cpp/topology/boundary_paths.hpp>

#include <cstdint>

int main() {
  using real_type = float;
  using index_type = std::int64_t;
  constexpr std::size_t dims = 2;
  const auto value =
      trueform_installed_test::open_triangle<index_type, real_type, dims>();

  const auto edges =
      tf::cpp::boundary_edges<index_type, real_type, dims>(value.mesh());
  const auto paths =
      tf::cpp::boundary_paths<index_type, real_type, dims>(value.mesh());
  const auto curves =
      tf::cpp::boundary_curves<index_type, real_type, dims>(value.mesh());
  return edges.raw_shape() == tf::small_vector<int, 3>{3, 2} &&
                 paths.size() == 1 && curves.paths.size() == 1 &&
                 curves.points.raw_shape() ==
                     tf::small_vector<int, 3>{3, static_cast<int>(dims)}
             ? 0
             : 1;
}
