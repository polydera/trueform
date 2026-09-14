#pragma once

#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/topology.hpp>

#include <cstddef>

namespace trueform_installed_test {

template <typename Index, typename Real, std::size_t Dims>
auto check_topology_analysis_axis() -> bool {
  const auto open = open_triangle<Index, Real, Dims>();
  if (tf::cpp::is_closed<Index, Real, Dims>(open.mesh()) ||
      !tf::cpp::is_open<Index, Real, Dims>(open.mesh()) ||
      !tf::cpp::is_manifold<Index, Real, Dims>(open.mesh()) ||
      tf::cpp::is_non_manifold<Index, Real, Dims>(open.mesh()) ||
      tf::cpp::non_manifold_edges<Index, Real, Dims>(open.mesh()).length() != 0)
    return false;

  const auto inconsistent = [] {
    if constexpr (Dims == 2)
      return consumer::owned_mesh<Index, Real, Dims, tf::dynamic_size>{
          polygons_of<Index, Real, Dims>({0, 3, 6}, {0, 1, 2, 1, 2, 3},
                                         {0, 0, 1, 0, 0, 1, 1, 1})};
    else
      return consumer::owned_mesh<Index, Real, Dims, tf::dynamic_size>{
          polygons_of<Index, Real, Dims>(
              {0, 3, 6}, {0, 1, 2, 1, 2, 3},
              {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0})};
  }();
  const auto oriented =
      tf::cpp::orient_faces_consistently<Index, Real, Dims>(
          inconsistent.mesh());
  if (oriented.size() != 2)
    return false;

  const auto faces = oriented.faces();
  const auto first = faces[0];
  const auto second = faces[1];
  auto shared_direction = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    const auto a = first[i];
    const auto b = first[(i + 1) % first.size()];
    for (std::size_t j = 0; j < second.size(); ++j)
      if (second[j] == b && second[(j + 1) % second.size()] == a)
        ++shared_direction;
  }
  return shared_direction == 1;
}

} // namespace trueform_installed_test
