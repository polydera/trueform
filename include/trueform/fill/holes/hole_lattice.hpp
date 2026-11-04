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
#include "../../core/point.hpp"
#include "../../exact/insphere.hpp"
#include "../../exact/meta.hpp"
#include "../hole_split.hpp"

namespace tf::fill {

/// The type a site coordinate is stored in: the input's lattice raised by
/// @ref tf::hole_split_depth bits of scale, which is a rung above the
/// lattice itself. The predicates keep dispatching on the lattice type and
/// read the scale through @ref tf::exact::insphere_scale_bits.
template <typename Int>
using hole_site_coord = typename tf::exact::meta<Int>::T1;

/// An original vertex's position on the scaled lattice.
template <typename Int>
auto hole_scaled_point(const tf::point<Int, 3> &p)
    -> tf::point<hole_site_coord<Int>, 3> {
  static_assert(tf::hole_split_depth <= tf::exact::insphere_scale_bits<Int>(),
                "the split scale must stay inside the predicates' headroom");
  using Coord = hole_site_coord<Int>;
  const Coord scale(tf::hole_split_scale);
  return {Coord(p[0]) * scale, Coord(p[1]) * scale, Coord(p[2]) * scale};
}

/// A scaled position back on the input's own lattice, which is exact for
/// every original vertex and for a split the scale can state.
template <typename Int>
auto hole_unscaled_point(const tf::point<hole_site_coord<Int>, 3> &p)
    -> tf::point<Int, 3> {
  using Coord = hole_site_coord<Int>;
  const Coord scale(tf::hole_split_scale);
  return {Int(p[0] / scale), Int(p[1] / scale), Int(p[2] / scale)};
}

/// The fill's lattice quantum: lattice units per unit of the mesh's own
/// coordinates, as the input's converter states it — its scale, or one where
/// the input already stands on its lattice.
template <typename Converter>
auto hole_lattice_unit(const Converter &converter)
    -> decltype(double(converter.scale)) {
  return double(converter.scale);
}

template <typename Converter, typename... Unused>
auto hole_lattice_unit(const Converter &, const Unused &...) -> double {
  return 1.0;
}

} // namespace tf::fill
