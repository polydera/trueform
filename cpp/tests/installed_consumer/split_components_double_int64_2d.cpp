#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/reindex/split_into_components.hpp>

#include <cstdint>
#include <iostream>
#include <type_traits>

int main() {
  using namespace trueform_installed_test;
  using real_type = double;
  using index_type = std::int64_t;
  constexpr auto dims = std::size_t{2};
  constexpr auto mixed = tf::dynamic_size;

  const consumer::owned_mesh<index_type, real_type, dims, mixed> dynamic_mesh{
      polygons_of<index_type, real_type, dims>(
          {0, 3, 6, 9, 12}, {0, 1, 2, 1, 3, 2, 3, 4, 5, 3, 5, 4},
          {0, 0, 1, 0, 0.5, 1, 2, 0, 2.5, 1, 3, 0})};
  const consumer::owned_edge_mesh<index_type, real_type, dims> edges{
      segments_of<index_type, real_type, dims>({0, 1, 1, 2, 3, 4},
                                               {0, 0, 1, 0, 1, 1, 2, 0, 3, 0})};

  const auto mesh_result = tf::cpp::split_into_components(
      dynamic_mesh.mesh(), make_array<std::int32_t>({0, 0, 1, 1}, {4}));
  const auto edge_result =
      tf::cpp::split_into_components<index_type, real_type, dims>(
          edges.edge_mesh(), make_array<std::int32_t>({0, 0, 1}, {3}));

  static_assert(
      std::is_same_v<std::decay_t<decltype(mesh_result)>,
                     tf::cpp::split_components_result<tf::polygons_buffer<
                         index_type, real_type, dims, mixed>>>);
  static_assert(
      std::is_same_v<std::decay_t<decltype(edge_result)>,
                     tf::cpp::split_components_result<
                         tf::segments_buffer<index_type, real_type, dims>>>);

  const auto ok =
      mesh_result.components.size() == 2 && mesh_result.labels.length() == 2 &&
      mesh_result.labels[0] == 0 && mesh_result.labels[1] == 1 &&
      mesh_result.components[0].size() == 2 &&
      mesh_result.components[1].size() == 2 &&
      mesh_result.components[0].points_buffer().size() == 4 &&
      mesh_result.components[1].points_buffer().size() == 3 &&
      mesh_result.components[0].faces_buffer().offsets_buffer()[2] ==
          index_type{6} &&
      edge_result.components.size() == 2 && edge_result.labels.length() == 2 &&
      edge_result.labels[0] == 0 && edge_result.labels[1] == 1 &&
      edge_result.components[0].size() == 2 &&
      edge_result.components[0].points_buffer().size() == 3 &&
      edge_result.components[1].size() == 1 &&
      edge_result.components[1].points_buffer().size() == 2;
  if (!ok)
    std::cerr << "double/int64/2D dynamic mesh or EdgeMesh split mismatch\n";
  return ok ? 0 : 1;
}
