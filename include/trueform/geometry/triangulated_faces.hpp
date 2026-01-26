/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once
#include "../core/algorithm/generic_generate.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/point.hpp"
#include "../core/polygons.hpp"
#include "../core/projector.hpp"
#include "../core/small_vector.hpp"
#include "./impl/ear_cutter.hpp"

namespace tf {
namespace impl {

template <typename Policy>
auto triangulated_faces_2d(const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;

  tf::buffer<Index> out;
  tf::generic_generate(
      polygons, out, tf::geom::earcutter<Index>{},
      [](const auto &poly, auto &buffer, auto &earcut) {
        earcut(poly);
        auto r = tf::make_indirect_range(earcut.indices(), poly.indices());
        std::copy(r.begin(), r.end(), std::back_inserter(buffer));
      });

  return tf::blocked_buffer<Index, 3>{std::move(out)};
}

template <typename Policy>
auto triangulated_faces_nd(const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;

  tf::buffer<Index> out;
  tf::generic_generate(
      polygons, out,
      std::make_pair(tf::small_vector<tf::point<double, 2>, 10>{},
                     tf::geom::earcutter<Index>{}),
      [](const auto &poly, auto &buffer, auto &state) {
        auto &[pts, earcut] = state;
        auto projector = tf::make_simple_projector(poly);
        pts.clear();
        for (const auto &v : poly)
          pts.push_back(projector(v));
        earcut(pts);
        auto r = tf::make_indirect_range(earcut.indices(), poly.indices());
        std::copy(r.begin(), r.end(), std::back_inserter(buffer));
      });

  return tf::blocked_buffer<Index, 3>{std::move(out)};
}

} // namespace impl

/// @ingroup geometry_processing
/// @brief Triangulate all polygons and return triangle indices.
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input polygons.
/// @return A blocked_buffer containing triangle indices (3 indices per
/// triangle).
template <typename Policy>
auto triangulated_faces(const tf::polygons<Policy> &polygons) {
  if constexpr (tf::coordinate_dims_v<Policy> == 2)
    return impl::triangulated_faces_2d(polygons);
  else
    return impl::triangulated_faces_nd(polygons);
}

} // namespace tf
