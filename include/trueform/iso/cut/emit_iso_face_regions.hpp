/*
 * Copyright (c) 2026 XLAB
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

#include "../../core/coordinate_dims.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../core/views/slide_range.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/plane_support.hpp"
#include "../../exact/vertex.hpp"
#include "./iso_band_of_triangles.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::iso {

/// The real position of a point the triangulation created. It is interior to
/// the face and no neighbour names it, so the corners the face's carrier
/// stands on are its authority: the lattice site is solved for the two
/// parameters that span the projection, and the real corners blend them.
/// Reached only when a build creates a point, so it finds those corners on the
/// spot rather than leaving a support beside every face that never needs one.
template <typename Int, typename Polygon, typename Converter, typename Site>
auto lift_iso_face_point(const Polygon &polygon, const std::pair<int, int> &axes,
                         const Converter &converter, const Site &site)
    -> tf::point<tf::coordinate_type<Polygon>, tf::coordinate_dims_v<Polygon>> {
  using RealT = tf::coordinate_type<Polygon>;
  tf::exact::plane_support<Int> support;
  std::array<std::size_t, 3> at{};
  const auto n = std::size_t(polygon.size());
  for (std::size_t c = 0; c < n && support.size < 3; ++c)
    if (support.offer(converter(polygon[c])))
      at[std::size_t(support.size - 1)] = c;
  std::array<tf::exact::pt2<Int>, 3> corners{};
  for (std::size_t c = 0; c < 3; ++c)
    corners[c] = tf::exact::pt2<Int>{support.point[c][axes.first],
                                     support.point[c][axes.second]};
  const double d1x = double(corners[1][0]) - double(corners[0][0]);
  const double d1y = double(corners[1][1]) - double(corners[0][1]);
  const double d2x = double(corners[2][0]) - double(corners[0][0]);
  const double d2y = double(corners[2][1]) - double(corners[0][1]);
  const double rx = double(site[0]) - double(corners[0][0]);
  const double ry = double(site[1]) - double(corners[0][1]);
  const double det = d1x * d2y - d2x * d1y;
  const double s = (rx * d2y - d2x * ry) / det;
  const double t = (d1x * ry - rx * d1y) / det;
  tf::point<RealT, tf::coordinate_dims_v<Polygon>> out;
  for (std::size_t d = 0; d < tf::coordinate_dims_v<Polygon>; ++d)
    out[d] = RealT(double(polygon[at[0]][d]) +
                   s * (double(polygon[at[1]][d]) - double(polygon[at[0]][d])) +
                   t * (double(polygon[at[2]][d]) - double(polygon[at[0]][d])));
  return out;
}

/// The winding a triangulation emits in its own projection, measured on the
/// first triangle whose doubled area is not zero.
template <typename Int, typename Faces, typename Points>
auto iso_cdt_orientation(const Faces &faces, const Points &points) -> int {
  for (std::size_t t = 0; t < std::size_t(faces.size()); ++t) {
    const auto triangle = faces[t];
    const auto &p0 = points[std::size_t(triangle[0])];
    const auto &p1 = points[std::size_t(triangle[1])];
    const auto &p2 = points[std::size_t(triangle[2])];
    const int sign = tf::exact::orient2d_sign(
        tf::exact::pt2<Int>{p0[0], p0[1]}, tf::exact::pt2<Int>{p1[0], p1[1]},
        tf::exact::pt2<Int>{p2[0], p2[1]});
    if (sign != 0)
      return sign;
  }
  return 0;
}

/// CORE. Read one cut face's triangulation back into the flat identity space
/// and publish its regions.
///
/// Every output point an input reached takes the MINIMAL flat identity that
/// reached it, so two faces sharing an edge resolve a lattice coincidence the
/// same way; an output no input reached is one the constraint recovery created
/// and takes a block-local ticket encoded as `-1 - k`.
///
/// A region lies strictly inside one band, so the first corner that is an
/// original off every cut value states it; a region with none is bounded by
/// chords and on-cut corners alone and takes the highest bounding cut index.
/// Region 0 is the hull exterior together with every non-convex pocket, and is
/// the only thing dropped.
template <typename LabelType, typename Index, typename Int, typename Local,
          typename Polygon, typename Converter, typename GetCategory,
          typename GetOnCut, typename GetCreatedCut>
auto emit_iso_face_regions(Local &local, Index object, const Polygon &polygon,
                           const std::pair<int, int> &axes,
                           const Converter &converter, Index n_original,
                           const GetCategory &get_category,
                           const GetOnCut &get_on_cut,
                           const GetCreatedCut &get_created_cut) -> void {
  const auto &triangles = local.cdt.faces_buffer();
  const auto n_triangles = std::size_t(triangles.size());
  if (n_triangles == 0)
    return;
  const auto labels = local.cdt.region_labels();
  const auto points = local.cdt.points();
  const auto forward = local.cdt.index_map().f();
  const auto n_final = std::size_t(points.size());

  local.rep.allocate(n_final);
  std::fill(local.rep.begin(), local.rep.end(), Index(-1));
  Index n_named = 0;
  for (Index input = 0; input < Index(local.chain.size()); ++input) {
    auto &representative = local.rep[std::size_t(forward[std::size_t(input)])];
    if (representative == Index(-1)) {
      representative = input;
      ++n_named;
    } else if (local.chain[std::size_t(input)] <
               local.chain[std::size_t(representative)])
      representative = input;
  }

  if (n_named != Index(n_final)) {
    local.mint_ticket.allocate(n_final);
    for (std::size_t out = 0; out < n_final; ++out) {
      if (local.rep[out] != Index(-1))
        continue;
      local.mint_ticket[out] = Index(local.mints.size());
      local.mints.push_back(
          lift_iso_face_point<Int>(polygon, axes, converter, points[out]));
    }
  }
  const auto corner_of = [&](Index point) -> Index {
    const auto representative = local.rep[std::size_t(point)];
    return representative == Index(-1)
               ? Index(-1) - local.mint_ticket[std::size_t(point)]
               : local.chain[std::size_t(representative)];
  };

  const bool flip =
      iso_cdt_orientation<Int>(triangles, points) * local.chain_orientation < 0;

  Index n_regions = 0;
  for (const auto label : labels)
    n_regions = std::max(n_regions, Index(label) + Index(1));
  local.region_offsets.allocate(std::size_t(n_regions) + 1);
  std::fill(local.region_offsets.begin(), local.region_offsets.end(), Index(0));
  for (const auto label : labels)
    ++local.region_offsets[std::size_t(label)];
  for (auto pair : tf::make_slide_range<2>(local.region_offsets))
    pair[1] += pair[0];
  local.region_order.allocate(n_triangles);
  for (std::size_t t = n_triangles; t-- > 0;)
    local.region_order[std::size_t(
        --local.region_offsets[std::size_t(labels[t])])] = Index(t);

  for (Index region = 1; region < n_regions; ++region) {
    const auto begin = local.region_offsets[std::size_t(region)];
    const auto end = local.region_offsets[std::size_t(region) + 1];
    if (begin == end)
      continue;
    const auto first = local.tris.size();
    local.tri_offsets.push_back(Index(first));
    for (auto k = begin; k < end; ++k) {
      const auto triangle = triangles[std::size_t(local.region_order[k])];
      std::array<Index, 3> corners{corner_of(triangle[0]),
                                   corner_of(triangle[1]),
                                   corner_of(triangle[2])};
      if (flip)
        std::swap(corners[1], corners[2]);
      local.tris.push_back(corners);
    }
    local.bands.push_back(iso_band_of_triangles<LabelType>(
        tf::make_range(local.tris.begin() + std::ptrdiff_t(first),
                       local.tris.end()),
        n_original, get_category, get_on_cut, get_created_cut));
    local.faces.push_back(object);
  }
}

} // namespace tf::iso
