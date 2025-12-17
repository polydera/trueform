/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/polygon.hpp"
#include "../core/polygons_buffer.hpp"
#include "./triangulated_faces.hpp"

namespace tf {

/// @brief Triangulate all polygons and return a triangle mesh buffer.
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input polygons.
/// @return A polygons_buffer containing triangulated mesh (3 indices per face).
template <typename Policy>
auto triangulated(const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using RealT = tf::coordinate_type<Policy>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy>;

  auto faces = tf::triangulated_faces(polygons);

  tf::polygons_buffer<Index, RealT, Dims, 3> out;
  out.faces_buffer() = std::move(faces);
  out.points_buffer().allocate(polygons.points().size());
  tf::parallel_copy(polygons.points(), out.points());

  return out;
}

template <std::size_t Dims, typename Policy>
auto triangulated(const tf::polygon<Dims, Policy> &polygon) {
  auto make_polygon_f = [&polygon](const auto &pts) {
    using RealT = tf::coordinate_type<Policy>;
    tf::polygons_buffer<int, RealT, Dims, 3> out;
    tf::geom::earcutter<int> earcut{};
    earcut(pts);
    std::copy(earcut.indices().begin(), earcut.indices().end(),
              std::back_inserter(out.faces_buffer().data_buffer()));

    out.points_buffer().allocate(polygon.size());
    tf::parallel_copy(polygon, out.points());
    return out;
  };
  if constexpr (Dims == 2)
    return make_polygon_f(polygon);
  else {
    tf::small_vector<tf::point<double, 2>, 10> pts{};
    auto projector = tf::make_simple_projector(polygon);
    for (const auto &v : polygon)
      pts.push_back(projector(v));
    return make_polygon_f(pts);
  }
}

} // namespace tf
