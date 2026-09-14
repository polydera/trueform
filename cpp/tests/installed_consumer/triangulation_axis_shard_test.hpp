#pragma once

#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/geometry/triangulate.hpp>

#include <cstddef>
#include <type_traits>

namespace trueform_installed_test {

template <typename Index, typename Real, std::size_t Dims>
auto check_triangulation_axis_shard() -> bool {
  using result_type = tf::polygons_buffer<Index, Real, Dims, 3>;
  auto points = quad_points<Real, Dims>();
  auto fixed_faces = make_array<Index>({0, 1, 2, 3}, {1, 4});
  auto dynamic_faces = dynamic_quad_faces<Index>();
  const auto value = quad_mesh<Index, Real, Dims>();

  const auto from_mesh =
      tf::cpp::triangulate<Index, Real, Dims>(value.mesh());
  const auto from_dynamic =
      tf::cpp::triangulate<Index, Real, Dims>(dynamic_faces, points);
  const auto from_fixed =
      tf::cpp::triangulate<Index, Real, Dims>(fixed_faces, points);
  const auto from_polygons = tf::cpp::triangulate<Index, Real, Dims>(points);
  static_assert(std::is_same_v<std::decay_t<decltype(from_mesh)>, result_type>);

  const auto is_quad_result = [](const auto &result) {
    return result.size() == 2 && result.points_buffer().size() == 4 &&
           consumer::face_indices_of(result).size() == 6;
  };
  return is_quad_result(from_mesh) && is_quad_result(from_dynamic) &&
         is_quad_result(from_fixed) && is_quad_result(from_polygons);
}

} // namespace trueform_installed_test
