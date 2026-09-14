#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/core/build_edge_membership.hpp>
#include <trueform/cpp/core/build_tree.hpp>
#include <trueform/cpp/core/build_vertex_link.hpp>

#include <cstdint>

int main() {
  const consumer::owned_edge_mesh<std::int64_t, float, 2> owned{
      trueform_installed_test::segments_of<std::int64_t, float, 2>(
          {0, 1, 1, 2, 1, 3}, {0, 0, 2, 0, 2, 3, -1, 0})};
  const auto value = owned.edge_mesh();

  tf::cpp::build_tree(value);
  tf::cpp::build_edge_membership(value);
  tf::cpp::build_vertex_link(value);
  const auto membership = value.edge_membership();
  const auto link = value.vertex_link();
  return value.number_of_edges() == 3 && value.number_of_points() == 4 &&
                 owned.cache.is_tree_fresh(value.geometry()) &&
                 owned.cache.is_edge_membership_fresh(value.geometry()) &&
                 owned.cache.is_vertex_link_fresh(value.geometry()) &&
                 value.tree().ids().size() == 3 && membership.size() == 4 &&
                 membership[1].size() == 3 && link.size() == 4 &&
                 link[1].size() == 3
             ? 0
             : 1;
}
