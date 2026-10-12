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

#include "../../exact/float_filter_sound.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/offset_dot.hpp"
#include "../../exact/offset_dot/filter_sign.hpp"
#include "../../exact/projected_dot.hpp"
#include "../../exact/projected_dot/filter_sign.hpp"
#include "../../exact/triple_product.hpp"
#include "../../exact/triple_product/filter_sign.hpp"
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// What the emitted half-planes say about a ring's order: `contradicted`, a
/// page turns back past one before it; `tied`, a page stands on its
/// predecessor's half-plane — two pages are two plane identities, and one
/// emitted half-plane cannot tell them apart; `flat`, a page stands on the
/// piece's own line, where it separates nothing; `astray`, a page's emitted
/// half-plane stands across another page's carrier from the side its own
/// carrier places it on. `tie` names the first two pages found tied.
struct plane_radial_ring_emission {
  bool contradicted = false;
  bool tied = false;
  bool flat = false;
  bool astray = false;
  std::array<std::size_t, 2> tie{};
};

/// The double image of an exact vector: each coordinate converted once, the
/// value every filter of the ring decides on.
template <typename V> auto plane_radial_ring_image(const V &v) {
  return std::array<double, 3>{double(v[0]), double(v[1]), double(v[2])};
}

/// A page's carrier, as the ring reads it: the plane's identity, its normal,
/// the side of it the page is, and the piece's start against the plane —
/// `normal . start - offset` — with the double images of the normal and the
/// residual.
template <typename Index, typename Int> struct plane_radial_ring_carrier {
  std::array<typename tf::exact::meta<Int>::T2, 3> normal;
  typename tf::exact::meta<Int>::T2 residual;
  std::array<double, 3> normal_image;
  double residual_image;
  Index plane;
  signed char side;
};

/// Read a ring's order against its emitted half-planes. `halves[r]` is page
/// `r`'s — its triangle's apex less the piece's start, in ring order — and
/// `halves_image[r]` its double image, `line` the piece's own run from start
/// to end, and `carriers[r]` page `r`'s carrier.
///
/// A triangle runs its piece with its apex on the left of its own winding,
/// which is its carrier's normal on its page's side, so page `j`'s
/// half-plane leaves the line along `wedge_j x line` and stands on the side
/// `line . (normal_i x wedge_j)` of carrier `i`; its emitted apex stands on
/// the side `normal_i . apex_j - offset_i`. Every sign is exact: the filters
/// decide on the images, which are the values they would convert, and the
/// exact rung answers what they leave undecided.
template <typename Int, typename Halves, typename HalvesImage,
          typename Carriers>
auto read_plane_radial_ring_emission(
    const Halves &halves, const HalvesImage &halves_image,
    const std::array<typename tf::exact::meta<Int>::T1, 3> &line,
    const Carriers &carriers) -> plane_radial_ring_emission {
  using T2 = typename tf::exact::meta<Int>::T2;
  using T3 = typename tf::exact::meta<Int>::T3;
  const auto line_image = plane_radial_ring_image(line);
  const auto turn = [&](std::size_t a, std::size_t b) {
    if constexpr (tf::exact::float_filter_sound) {
      const int filtered = tf::exact::triple_product::filter_sign(
          halves_image[a], halves_image[b], line_image);
      if (filtered)
        return filtered;
    }
    return tf::exact::triple_product_sign<T2>(halves[a], halves[b], line);
  };
  // the sign of the two half-planes' dot once the line is projected out
  const auto sense = [&](std::size_t a, std::size_t b) {
    if constexpr (tf::exact::float_filter_sound) {
      const int filtered = tf::exact::projected_dot::filter_sign(
          halves_image[a], halves_image[b], line_image);
      if (filtered)
        return filtered;
    }
    return tf::exact::projected_dot_sign<T3>(halves[a], halves[b], line);
  };

  plane_radial_ring_emission emission;
  bool seen = false;
  std::size_t first = 0, previous = 0;
  int previous_quadrant = 0;
  for (std::size_t r = 0; r < std::size_t(halves.size()); ++r) {
    if (sense(r, r) == 0) {
      emission.flat = true;
      continue;
    }
    if (!seen) {
      seen = true;
      first = previous = r;
      continue;
    }
    // the angle from the first page: 0 on it, 1 short of a half turn, 2 at
    // it, 3 past it
    const int t0 = turn(first, r);
    const int q = t0 != 0 ? (t0 > 0 ? 1 : 3) : (sense(first, r) > 0 ? 0 : 2);
    const int t = q == previous_quadrant ? turn(previous, r) : 1;
    if (q < previous_quadrant || t < 0) {
      emission.contradicted = true;
      return emission;
    }
    if (t == 0 && !emission.tied) {
      emission.tied = true;
      emission.tie = {previous, r};
    }
    previous = r;
    previous_quadrant = q;
  }

  // the side of carrier `a` the line turns carrier `b`'s normal to
  const auto across = [&](std::size_t a, std::size_t b) {
    if constexpr (tf::exact::float_filter_sound) {
      const int filtered = tf::exact::triple_product::filter_sign(
          carriers[a].normal_image, carriers[b].normal_image, line_image);
      if (filtered)
        return filtered;
    }
    return tf::exact::triple_product_sign<T3>(carriers[a].normal,
                                              carriers[b].normal, line);
  };
  // the side of carrier `a` page `b`'s emitted apex stands on
  const auto stands = [&](std::size_t a, std::size_t b) {
    if constexpr (tf::exact::float_filter_sound) {
      const int filtered = tf::exact::offset_dot::filter_sign(
          carriers[a].normal_image, halves_image[b],
          carriers[a].residual_image);
      if (filtered)
        return filtered;
    }
    return tf::exact::offset_dot_sign<T2>(carriers[a].normal, halves[b],
                                          carriers[a].residual);
  };
  const auto disagree = [](int placed, int stood) {
    return placed != 0 && stood != 0 && placed != stood;
  };
  const auto n_pages = std::size_t(carriers.size());
  for (std::size_t i = 0; i < n_pages; ++i)
    for (std::size_t j = i + 1; j < n_pages; ++j) {
      if (carriers[i].plane == carriers[j].plane)
        continue;
      const int turned = across(i, j);
      if (disagree(turned * carriers[j].side, stands(i, j)) ||
          disagree(-turned * carriers[i].side, stands(j, i))) {
        emission.astray = true;
        return emission;
      }
    }
  return emission;
}

} // namespace tf::csg::graph
