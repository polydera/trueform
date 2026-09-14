#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/clean/edge_mesh.hpp>
#include <trueform/cpp/clean/mesh.hpp>

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
          {0, 3, 7}, {0, 1, 2, 3, 1, 4, 5},
          {0, 0, 1, 0, 0.5, 1, 0, 0, 1.5, 0.5, 9, 9})};
  const consumer::owned_edge_mesh<index_type, real_type, dims> edges{
      segments_of<index_type, real_type, dims>(
          {0, 1, 2, 3}, {0, 0, 1, 0, 0.5, 1, 0, 0, 1.5, 0.5, 9, 9})};

  const auto cleaned_dynamic =
      tf::cpp::cleaned_mesh_with_maps(dynamic_mesh.mesh());
  const auto direct_dynamic = tf::cpp::cleaned_mesh(dynamic_mesh.mesh());
  const auto cleaned_edges =
      tf::cpp::cleaned_edge_mesh_with_maps<index_type, real_type, dims>(
          edges.edge_mesh());
  const auto direct_edges =
      tf::cpp::cleaned_edge_mesh<index_type, real_type, dims>(
          edges.edge_mesh());

  static_assert(
      std::is_same_v<
          std::decay_t<decltype(cleaned_dynamic)>,
          tf::cpp::cleaned_mesh_result<index_type, real_type, dims, mixed>>);
  static_assert(
      std::is_same_v<
          std::decay_t<decltype(cleaned_edges)>,
          tf::cpp::cleaned_edge_mesh_result<index_type, real_type, dims>>);
  static_assert(
      std::is_same_v<std::decay_t<decltype(cleaned_dynamic.face_map.f)>,
                     tf::cpp::nd_array<index_type>>);

  const auto &offsets = cleaned_dynamic.mesh.faces_buffer().offsets_buffer();
  const auto ok =
      cleaned_dynamic.mesh.size() == 2 &&
      cleaned_dynamic.mesh.points_buffer().size() == 5 && offsets.size() == 3 &&
      offsets[1] == index_type{3} && offsets[2] == index_type{7} &&
      cleaned_dynamic.face_map.f.length() == 2 &&
      cleaned_dynamic.point_map.f.length() == 6 &&
      cleaned_dynamic.point_map.kept_ids.length() == 5 &&
      cleaned_dynamic.point_map.f[0] == cleaned_dynamic.point_map.f[3] &&
      direct_dynamic.size() == 2 &&
      direct_dynamic.points_buffer().size() == 5 &&
      cleaned_edges.edge_mesh.size() == 2 &&
      cleaned_edges.edge_mesh.points_buffer().size() == 3 &&
      cleaned_edges.edge_map.f.length() == 2 &&
      cleaned_edges.point_map.f.length() == 6 &&
      cleaned_edges.point_map.f[0] == cleaned_edges.point_map.f[3] &&
      direct_edges.size() == 2 && direct_edges.points_buffer().size() == 3;
  if (!ok)
    std::cerr << "double/int64/2D dynamic mesh or EdgeMesh cleaning mismatch\n";
  return ok ? 0 : 1;
}
