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
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../exact_coordinate_converter.hpp"
#include "./delaunay_topology_owner.hpp"
#include "./delaunay_topology_policy.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tf::topology::cdt {

/// Authoritative flat state for the point-only Delaunay tier.
template <typename Index, typename Coord, typename Int, typename VertexPolicy>
struct unconstrained_delaunay_owner
    : delaunay_topology_owner<Index, Int,
                              triangle_only_topology_policy<Index>> {
  using base_type =
      delaunay_topology_owner<Index, Int, triangle_only_topology_policy<Index>>;
  using index_type = Index;
  using coord_type = Coord;
  using int_type = Int;
  using vertex_policy = VertexPolicy;

  static constexpr Index none = std::numeric_limits<Index>::max();

  tf::blocked_buffer<Index, 3> _faces;
  tf::buffer<std::size_t> _face_offsets;
  tf::buffer<std::uint8_t> _face_visited;
  tf::exact_coordinate_converter<Int, Coord, 2> _converter;
};

} // namespace tf::topology::cdt
