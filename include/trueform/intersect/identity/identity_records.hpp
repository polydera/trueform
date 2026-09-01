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
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../exact/edge_parameter.hpp"

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::intersect {

/// The original vertex a kind-V point is anchored at.
template <typename Index> struct vertex_anchor {
  std::int16_t tag;
  Index vid;
};

/// A kind-E carrier, named by its two canonical flat vertex ids with
/// `u < v`. Tag-free: instances of one physical edge — duplicated
/// vertices within a form, an edge shared across forms — carry
/// different raw ids but one canonical key, so they are one carrier by
/// construction.
template <typename Index> struct home_edge {
  Index u, v;

  friend auto operator<(const home_edge &a, const home_edge &b) -> bool {
    return std::make_pair(a.u, a.v) < std::make_pair(b.u, b.v);
  }

  friend auto operator==(const home_edge &a, const home_edge &b) -> bool {
    return a.u == b.u && a.v == b.v;
  }
};

/// Record `row` sits at canonical vertex `vertex`.
template <typename Index> struct vertex_incidence {
  Index vertex;
  Index row;
};

/// An edge target resolved onto the canonical vertex space: the carrier
/// it names, its own two endpoints as flat vertex ids, and whether the
/// carrier runs against the direction a parameter on this edge is
/// stated in — from the edge's lower flat id, the convention the
/// emitting kernel wrote its fraction in.
template <typename Index> struct edge_instance {
  home_edge<Index> carrier;
  Index from, to;
  bool reversed;
};

/// Record `row` sits on `carrier`. `slot` is which of the record's two
/// identity slots the resulting class fills — a record that also has a
/// vertex target keeps slot 0 for the vertex, so its carrier class lands
/// in slot 1 and the two are declared equivalent. `parameter` is which
/// of the record's two stored fractions belongs to this carrier, and
/// `reversed` whether it must be restated in the carrier's direction.
template <typename Index> struct edge_candidate {
  home_edge<Index> carrier;
  Index row;
  std::int8_t slot;
  std::int8_t parameter;
  std::int8_t reversed;

  friend auto operator<(const edge_candidate &a, const edge_candidate &b)
      -> bool {
    return std::make_tuple(a.carrier.u, a.carrier.v, a.row, a.parameter) <
           std::make_tuple(b.carrier.u, b.carrier.v, b.row, b.parameter);
  }
};

/// Per-carrier exact-parameter classes. `offsets` blocks the class
/// arrays by carrier, and within a block the classes ascend in exact
/// parameter — the order @ref tf::polygon_intersections publishes as
/// its split CSR.
template <typename Index, typename Int> struct edge_classes {
  tf::buffer<home_edge<Index>> carriers;
  tf::buffer<Index> offsets;
  tf::buffer<Index> endpoint;
  tf::buffer<tf::exact::edge_parameter<Int>> parameter;
};

/// Reusable capacity for identity formation: held by the class, so a
/// rebuilt instance pays allocation once, not per build. Contents are
/// transient — every user clears or overwrites what it reads.
template <typename Index, typename Int> struct identity_scratch {
  tf::buffer<vertex_incidence<Index>> incidences;
  tf::buffer<edge_candidate<Index>> candidates;
  tf::buffer<Index> candidate_class;
  edge_classes<Index, Int> classes;
  tf::buffer<Index> vertex_keys;
  tf::buffer<Index> row_class;
  tf::buffer<std::array<Index, 2>> pairs;
  tf::buffer<Index> point_of;
  tf::buffer<Index> carrier_offsets;
  tf::buffer<Index> slot_endpoint;
  tf::buffer<tf::exact::edge_parameter<Int>> slot_parameter;
  tf::buffer<Index> counts;
};

/// The canonical point tables: kind-V ids index `vertex_anchors`,
/// kind-E ids index `home_edges` / `parameters` after subtracting
/// `vertex_anchors.size()`.
template <typename Index, typename Int> struct point_identities {
  tf::buffer<vertex_anchor<Index>> vertex_anchors;
  tf::buffer<home_edge<Index>> home_edges;
  tf::buffer<tf::exact::edge_parameter<Int>> parameters;
  tf::buffer<home_edge<Index>> edge_carriers;
  tf::offset_block_buffer<Index, Index> edge_splits;

  auto clear() -> void {
    vertex_anchors.clear();
    home_edges.clear();
    parameters.clear();
    edge_carriers.clear();
    edge_splits.clear();
  }
};

} // namespace tf::intersect
