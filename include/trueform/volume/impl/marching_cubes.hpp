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
#include "../../clean/soup/polygons.hpp"
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../volume.hpp"
#include "./field_value.hpp"
#include "./marching_cubes_tables.hpp"
#include <cstddef>
#include <tuple>
#include <utility>

namespace tf {
namespace volume_detail {

/// @brief Marching-cubes isosurface extractor (Lorensen & Cline).
///
/// The correct-and-simple baseline: emits a triangle soup and welds coincident
/// edge vertices into a shared-vertex indexed mesh. A corner is inside when
/// `sample < iso` (outward winding for an SDF). Kept as the reference against
/// which the `flying_edges` fast path is verified; `tf::make_isosurface` uses
/// the latter.
template <typename Index = int, typename Real, typename Policy>
auto marching_cubes(const tf::volume<Policy> &vol, Real iso)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  const auto &dims = vol.dims();
  if (dims[0] < 2 || dims[1] < 2 || dims[2] < 2)
    return {};

  // Sample the 8 cube corners at cell (x, y, z) and return the inside-mask.
  auto classify_cell = [&](int x, int y, int z, Real corner_value[8]) -> int {
    int cube_index = 0;
    for (int i = 0; i < 8; ++i) {
      const auto &o = mc_corner_offsets[i];
      const Real v = field_value<Real>(vol(x + o[0], y + o[1], z + o[2]));
      corner_value[i] = v;
      if (v < iso)
        cube_index |= (1 << i);
    }
    return cube_index;
  };

  // Pass 1 — count the triangles so the soup buffer can be sized exactly.
  std::size_t triangle_count = 0;
  for (int z = 0; z < dims[2] - 1; ++z)
    for (int y = 0; y < dims[1] - 1; ++y)
      for (int x = 0; x < dims[0] - 1; ++x) {
        Real corner_value[8];
        const int cube_index = classify_cell(x, y, z, corner_value);
        const int *row = mc_tri_table[cube_index];
        for (int e = 0; row[e] != -1; e += 3)
          ++triangle_count;
      }

  if (triangle_count == 0)
    return {};

  // Interpolate the surface crossing on cube edge `e` of cell (x, y, z).
  //
  // The two endpoints are ordered canonically by their global (z, y, x) index
  // before interpolating. A physical grid edge is shared by up to four cells,
  // each numbering it from a different local corner; without canonicalization
  // the crossing point differs by a few ULPs between neighbours and the exact
  // weld leaves hairline cracks. Fixing the endpoint order makes every cell
  // produce a bit-identical point for a shared edge, so the surface is exactly
  // watertight.
  auto edge_point = [&](int x, int y, int z, const Real corner_value[8],
                        int e) -> tf::point<Real, 3> {
    const int a = mc_edge_corners[e][0];
    const int b = mc_edge_corners[e][1];
    const auto &oa = mc_corner_offsets[a];
    const auto &ob = mc_corner_offsets[b];
    int ax = x + oa[0], ay = y + oa[1], az = z + oa[2];
    int bx = x + ob[0], by = y + ob[1], bz = z + ob[2];
    Real va = corner_value[a];
    Real vb = corner_value[b];
    if (std::tie(az, ay, ax) > std::tie(bz, by, bx)) {
      std::swap(ax, bx);
      std::swap(ay, by);
      std::swap(az, bz);
      std::swap(va, vb);
    }
    const auto pa = vol.template point_at<Real>(ax, ay, az);
    const auto pb = vol.template point_at<Real>(bx, by, bz);
    const Real denom = vb - va;
    const Real t = (denom != Real{0}) ? (iso - va) / denom : Real{0.5};
    return tf::point<Real, 3>{pa[0] + (pb[0] - pa[0]) * t,
                              pa[1] + (pb[1] - pa[1]) * t,
                              pa[2] + (pb[2] - pa[2]) * t};
  };

  // Pass 2 — emit triangle soup (9 coordinates per triangle).
  tf::buffer<Real> soup;
  soup.allocate(triangle_count * 9);
  std::size_t w = 0;
  auto append_point = [&](const tf::point<Real, 3> &p) {
    soup[w++] = p[0];
    soup[w++] = p[1];
    soup[w++] = p[2];
  };

  for (int z = 0; z < dims[2] - 1; ++z)
    for (int y = 0; y < dims[1] - 1; ++y)
      for (int x = 0; x < dims[0] - 1; ++x) {
        Real corner_value[8];
        const int cube_index = classify_cell(x, y, z, corner_value);
        const int *row = mc_tri_table[cube_index];
        for (int e = 0; row[e] != -1; e += 3) {
          append_point(edge_point(x, y, z, corner_value, row[e]));
          append_point(edge_point(x, y, z, corner_value, row[e + 1]));
          append_point(edge_point(x, y, z, corner_value, row[e + 2]));
        }
      }

  // Weld coincident edge vertices into a shared-vertex indexed mesh.
  tf::clean::polygon_soup<Index, Real, 3, 3> welded;
  welded.build(std::move(soup));
  return welded;
}

} // namespace volume_detail
} // namespace tf
