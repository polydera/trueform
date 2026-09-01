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
#include "../../core/point.hpp"
#include "../../core/points_buffer.hpp"
#include "../../exact/incircle.hpp"
#include "../../exact/int32.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact_coordinate_converter.hpp"
#include "../cdt_region_mode.hpp"
#include "./constrained_delaunay_constraint_provenance.hpp"
#include "./delaunay_topology_owner.hpp"
#include "./delaunay_topology_policy.hpp"
#include "./retained_half_edges.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt {

/// Authoritative flat state for constrained Delaunay construction. The public
/// triangulator owns this carrier; free operations in this directory mutate it
/// phase by phase.
template <typename Index, typename Coord, typename Int,
          typename ExecutionPolicy>
struct constrained_delaunay_owner
    : delaunay_topology_owner<Index, Int, retained_topology_policy<Index>> {
  using base_type =
      delaunay_topology_owner<Index, Int, retained_topology_policy<Index>>;
  using index_type = Index;
  using coord_type = Coord;
  using int_type = Int;
  using execution_policy = ExecutionPolicy;
  using param_type = typename tf::exact::meta<Int>::param_type;
  using half_edge = retained_delaunay_half_edge<Index>;

  static constexpr Index none = Index(-1);
  static constexpr int crossing_param_bits = tf::exact::meta<Int>::param_bits;
  static constexpr int max_resolution_depth = 64;
  static constexpr std::uint8_t unconstrained = 0;
  static constexpr std::uint8_t constrained = 1;
  static constexpr std::uint8_t boundary_constrained = 2;

  using constraint_provenance =
      typename constrained_delaunay_constraint_provenance<
          Index, param_type>::value_type;

  struct incremental_crossing {
    Index point;
    Index id_a;
    Index id_b;
    param_type t_a;
    param_type t_b;
  };

  /// Two input constraints claimed one edge: coincidence over a span, not
  /// a crossing. Provenance keeps the later input; this row keeps the fact.
  struct constraint_collision {
    Index edge;
    Index prior_input;
    Index input;
  };

  struct flip_check {
    Index e = none;
    Index v0 = none;
    Index v1 = none;
    Index opposite_vertex = none;
  };

  enum class locate_kind { interior, boundary_edge, exterior };

  struct locate_result {
    locate_kind kind;
    Index he;
  };

  enum class obstruction_kind { none, crossing, vertex_on, collinear };

  struct obstruction {
    obstruction_kind kind = obstruction_kind::none;
    Index edge = none;
    Index vertex = none;
  };

  static auto opposite(Index edge) -> Index { return edge ^ Index(1); }

  auto next_edge(Index edge) const -> Index {
    return this->_edges[std::size_t(edge)].next;
  }

  auto previous_edge(Index edge) const -> Index {
    return this->_edges[std::size_t(edge)].prev;
  }

  auto origin(Index edge) const -> Index {
    return this->_edges[std::size_t(edge)].vertex;
  }

  auto target(Index edge) const -> Index { return origin(opposite(edge)); }

  auto output_vertex(Index topology_vertex) const -> Index {
    if (topology_vertex == none || this->_sites.size() == 0)
      return topology_vertex;
    return this->_sites[std::size_t(topology_vertex)].output;
  }

  auto point(Index vertex) const -> tf::point<Int, 2> {
    return tf::make_point(static_cast<Int>(_points[std::size_t(vertex)][0]),
                          static_cast<Int>(_points[std::size_t(vertex)][1]));
  }

  auto orient(Index a, Index b, Index c) const -> int {
    return tf::exact::orient2d_sign(point(a), point(b), point(c));
  }

  auto orient(Index a, Index b, const tf::point<Int, 2> &c) const -> int {
    return tf::exact::orient2d_sign(point(a), point(b), c);
  }

  auto incircle(Index a, Index b, Index c, Index d) const -> int {
    return tf::exact::incircle_sign(point(a), point(b), point(c), point(d));
  }

  tf::points_buffer<Int, 2> _points;
  tf::buffer<Index> _v_first_edge;

  tf::buffer<flip_check> _flip_stack;
  tf::buffer<Index> _deleted_edges;
  tf::buffer<Index> _vertices_cw;
  tf::buffer<Index> _vertices_ccw;
  tf::buffer<Index> _retriangulation_stack;
  tf::buffer<Index> _constraint_input_ids;
  bool _allow_crossing_splits = false;
  bool _track_constraint_provenance = false;
  constrained_delaunay_constraint_provenance<Index, param_type>
      _constraint_provenance;
  tf::buffer<incremental_crossing> _incremental_crossings;
  tf::buffer<constraint_collision> _constraint_collisions;
  tf::buffer<std::array<Index, 2>> _final_constraint_edges;
  tf::buffer<std::uint8_t> _final_is_boundary;
  bool _has_prepared_input_index_map = false;
  bool _always_track_constraint_provenance = false;

  Index _last_edge = none;
  tf::cdt_region_mode _region_mode = tf::cdt_region_mode::nesting;

  tf::exact_coordinate_converter<Int, Coord, 2> _converter;
  tf::blocked_buffer<Index, 3> _faces;
  tf::buffer<Index> _first_edge_of_triangle;
  tf::buffer<Index> _triangle_of_edge;
  tf::buffer<Index> _region_labels;

  tf::buffer<std::size_t> _scratch_face_offsets;
  tf::buffer<Index> _scratch_stack;

  Index _locate_hint = none;
};

} // namespace tf::topology::cdt
