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

#include "../clean/points.hpp"
#include "../core/algorithm/compose_index_maps.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_iota.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/edges.hpp"
#include "../core/index_map.hpp"
#include "../core/point.hpp"
#include "../core/points.hpp"
#include "../core/points_buffer.hpp"
#include "../core/views/constant.hpp"
#include "../core/views/mapped_range.hpp"
#include "../exact/incircle.hpp"
#include "../exact/int32.hpp"
#include "../exact/orient2d.hpp"
#include "../exact_coordinate_converter.hpp"
#include "../intersect/exact/segment_intersection_graph.hpp"
#include "../intersect/graph/vertex.hpp"
#include "../intersect/intersections_within_segments.hpp"
#include "../spatial/aabb_tree.hpp"
#include "../spatial/tree_config.hpp"
#include "./edge_membership.hpp"
#include "./topo_type.hpp"
#include <array>
#include <cstdint>

namespace tf {

/// @ingroup topology
/// @brief Constrained Delaunay triangulation of 2D point sets.
///
/// Builds an exact-predicate Delaunay triangulation of the cleaned input
/// points, arranges intersecting constraint edges, recovers the constraints,
/// and labels the resulting planar regions.
///
/// Input points are converted to `Int`, cleaned once after constraint
/// arrangement, and exposed through `points()`. Constraint intersections may
/// therefore introduce additional output points.
///
/// @tparam Index The index type.
/// @tparam Coord The input coordinate type. Also the type returned by
///   `points()` after deconversion. May be either an integral or a
///   floating-point type (default: float).
/// @tparam Int The integer type for exact predicates (default: int32).
template <typename Index, typename Coord, typename Int = tf::exact::int32>
class constrained_delaunay_triangulator {
  static constexpr Index k_none = Index(-1);

  // Constraint state of a half-edge.
  //   k_unconstrained        — flippable, ignored by region labelling.
  //   k_constrained          — preserved by Delaunay restoration; does
  //                            NOT separate regions (parity passes
  //                            through unchanged).
  //   k_boundary_constrained — preserved AND separates regions
  //                            (parity flips when crossed).
  static constexpr std::uint8_t k_unconstrained = 0;
  static constexpr std::uint8_t k_constrained = 1;
  static constexpr std::uint8_t k_boundary_constrained = 2;

  struct half_edge {
    Index vertex = k_none;
    Index prev = k_none;
    Index next = k_none;
    bool boundary = false;
    bool delaunay = false;
    std::uint8_t constrained = k_unconstrained;
  };

  struct flip_check {
    Index e = k_none;
    Index v0 = k_none;
    Index v1 = k_none;
    Index opposite_vertex = k_none;
  };

public:
  auto clear() -> void {
    _points = tf::points_buffer<Int, 2>{};
    _edges.clear();
    _v_first_edge.clear();
    _flip_stack.clear();
    _deleted_edges.clear();
    _vertices_cw.clear();
    _vertices_ccw.clear();
    _retriangulation_stack.clear();
    _constraint_edges.clear();
    _arranged_constraint_edges.clear();
    _final_constraint_edges.clear();
    _input_is_boundary.clear();
    _arranged_is_boundary.clear();
    _final_is_boundary.clear();
    _split_constraint_edge.clear();
    _augmented_points.clear();
    _constraint_em.offsets_buffer().clear();
    _constraint_em.data_buffer().clear();
    _constraint_tree.clear();
    _constraint_si.clear();
    _constraint_sig.clear();
    _last_edge = k_none;
    _n_triangles = Index(0);
    _region_labels.clear();
    _index_map.f().clear();
    _index_map.kept_ids().clear();
  }

  /// @brief Output points indexed by `make_faces()`.
  ///
  /// Returned coordinates are deconverted from the internal integer
  /// representation back to `Coord`.
  auto points() const { return tf::make_points(_points); }

  auto converted_points() const {
    return tf::make_points(tf::make_mapped_range(
        _points, [this](const auto &p) { return _converter.deconvert(p); }));
  }

  /// @brief Build and return triangles of the constrained triangulation.
  auto make_faces() const -> tf::blocked_buffer<Index, 3> {
    tf::blocked_buffer<Index, 3> out;
    out.reserve(static_cast<std::size_t>(_n_triangles));

    auto n_he = static_cast<Index>(_edges.size());

    tf::buffer<bool> checked;
    checked.allocate(n_he);
    tf::parallel_fill(checked, false);

    for (Index e0 = 0; e0 < n_he; ++e0) {
      if (checked[e0] || _edges[e0].boundary)
        continue;

      Index e1 = prev_e(opp(e0));
      Index e2 = prev_e(opp(e1));

      if (e1 == k_none || e2 == k_none)
        continue;

      if (_edges[e1].boundary || _edges[e2].boundary)
        continue;

      checked[e0] = true;
      checked[e1] = true;
      checked[e2] = true;

      out.push_back({_edges[e0].vertex, _edges[e1].vertex, _edges[e2].vertex});
    }

    return out;
  }

  /// @brief Index map between original input point space and the
  /// cleaned output point space.
  ///
  /// `f()[input_id]` is the output index of input point `input_id`.
  /// `kept_ids()[output_id]` is the original input ID that survived the
  /// arrangement and dedup for output slot `output_id`, or the sentinel
  /// `f().size()` for a synthetic intersection vertex.
  auto index_map() const -> const tf::index_map_buffer<Index> & {
    return _index_map;
  }

  /// @brief Non-const accessor to allow moving the index map out.
  auto index_map() -> tf::index_map_buffer<Index> & { return _index_map; }

  /// @brief Region labels separated by constrained edges, indexed by
  /// triangle (matches the triangle order produced by `make_faces()`).
  ///
  /// Labels are parity values propagated across triangles. Crossing a
  /// constrained edge toggles the label. Computed during `build()`.
  auto region_labels() const { return tf::make_range(_region_labels); }

  /// @brief Build the triangulation and recover the supplied constraints.
  ///
  /// Intersecting constraints are split before triangulation. The resulting
  /// output indices refer to `points()`, not necessarily the original input
  /// point IDs.
  ///
  /// `is_boundary` is a per-input-edge mask. An entry of `true` marks the
  /// constraint as a region boundary — `region_labels()` parity flips
  /// when crossed. `false` marks it as preserved-but-not-boundary
  /// (parity passes through). Pass `tf::make_constant_range(true,
  /// edges.size())` for the all-boundary case.
  template <typename PointsPolicy, typename EdgesPolicy,
            typename IsBoundaryRange>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges,
             const IsBoundaryRange &is_boundary) -> bool {
    clear();

    _converter = tf::make_exact_coordinate_converter<Int, Coord>(pts);

    auto int_pts = tf::make_points(tf::make_mapped_range(
        pts, [this](const auto &p) { return _converter(p); }));

    make_arranged_constraints(edges, is_boundary, int_pts);

    auto n = static_cast<Index>(_points.size());
    if (n < 3)
      return true;

    _v_first_edge.allocate(n);
    tf::parallel_fill(_v_first_edge, k_none);

    _edges.reserve(static_cast<std::size_t>(6 * n));
    _flip_stack.reserve(static_cast<std::size_t>(6 * n));

    if (!build_delaunay(n))
      return false;

    for (Index i = 0; i < static_cast<Index>(_final_constraint_edges.size());
         ++i) {
      auto edge = _final_constraint_edges[i];
      Index a = edge[0];
      Index b = edge[1];
      if (a == b)
        continue;
      if (!constrain_single_edge(a, b, _final_is_boundary[i] != 0))
        return false;
    }

    delaunay_flip();
    compute_region_labels();
    return true;
  }

  template <typename PointsPolicy, typename EdgesPolicy>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges) -> bool {
    return build(pts, edges, tf::make_constant_range(true, edges.size()));
  }

private:
  // The arrangement space is `[0, n_input + sig.points().size())`:
  // `[0, n_input)` are the original input points (identity), and the
  // tail is the SIG-created intersection points. The forward map is
  // identity over the original portion. The kept_ids carry the
  // original input ID for each arrangement slot — directly for the
  // identity portion, and from SI vertex records for SIG-created V
  // intersections (slots that geometrically coincide with an original
  // input vertex). EE intersection slots are marked with the sentinel
  // `n_input`, matching the @ref tf::update_by_mask convention.
  auto build_original_to_arrangement_index_map(Index n_input)
      -> tf::index_map_buffer<Index> {
    auto n_sig = static_cast<Index>(_constraint_sig.points().size());

    tf::index_map_buffer<Index> im;
    im.f().allocate(n_input);
    tf::parallel_iota(im.f(), Index(0));

    im.kept_ids().allocate(static_cast<std::size_t>(n_input + n_sig));
    tf::parallel_fill(im.kept_ids(), n_input);
    tf::parallel_iota(
        tf::make_range(im.kept_ids().begin(), im.kept_ids().begin() + n_input),
        Index(0));

    for (auto group : _constraint_si.intersections()) {
      for (const auto &rec : group) {
        if (rec.target.label != tf::topo_type::vertex)
          continue;
        Index original = _constraint_edges[rec.object][rec.target.id];
        Index sig_pt = _constraint_sig.point_remap()[rec.id];
        im.kept_ids()[n_input + sig_pt] = original;
      }
    }

    return im;
  }

  auto compute_region_labels() -> void {
    _region_labels.clear();
    auto n_he = static_cast<Index>(_edges.size());
    if (n_he == 0)
      return;

    tf::buffer<Index> tri_of_edge;
    tri_of_edge.allocate(n_he);
    tf::parallel_fill(tri_of_edge, k_none);

    tf::buffer<Index> first_edge_of_tri;
    first_edge_of_tri.reserve(static_cast<std::size_t>(_n_triangles));
    tf::buffer<bool> checked;
    checked.allocate(n_he);
    tf::parallel_fill(checked, false);

    for (Index e0 = 0; e0 < n_he; ++e0) {
      if (checked[e0] || _edges[e0].boundary) {
        if (_edges[e0].boundary)
          checked[e0] = true;
        continue;
      }

      Index e1 = prev_e(opp(e0));
      Index e2 = prev_e(opp(e1));
      if (e1 == k_none || e2 == k_none || _edges[e1].boundary ||
          _edges[e2].boundary)
        continue;

      Index tri = static_cast<Index>(first_edge_of_tri.size());
      first_edge_of_tri.push_back(e0);
      tri_of_edge[e0] = tri;
      tri_of_edge[e1] = tri;
      tri_of_edge[e2] = tri;
      checked[e0] = true;
      checked[e1] = true;
      checked[e2] = true;
    }

    _region_labels.allocate(first_edge_of_tri.size());
    tf::parallel_fill(_region_labels, k_none);
    if (first_edge_of_tri.size() == 0)
      return;

    Index start_tri = k_none;
    Index start_edge = k_none;
    for (Index e = 0; e < n_he; ++e) {
      if (_edges[e].boundary) {
        start_tri = tri_of_edge[opp(e)];
        if (start_tri != k_none) {
          start_edge = e;
          break;
        }
      }
    }
    if (start_tri == k_none)
      start_tri = Index(0);

    auto is_region_boundary = [&](Index e) {
      return _edges[e].constrained == k_boundary_constrained;
    };

    tf::buffer<Index> stack;
    stack.push_back(start_tri);
    _region_labels[start_tri] =
        start_edge != k_none && is_region_boundary(start_edge) ? Index(1)
                                                               : Index(0);

    while (stack.size() > 0) {
      Index tri = stack.back();
      stack.pop_back();

      Index e0 = first_edge_of_tri[tri];
      std::array<Index, 3> tri_edges{e0, prev_e(opp(e0)),
                                     prev_e(opp(prev_e(opp(e0))))};

      for (Index e : tri_edges) {
        Index neighbor = tri_of_edge[opp(e)];
        if (neighbor == k_none || _region_labels[neighbor] != k_none)
          continue;

        _region_labels[neighbor] =
            _region_labels[tri] ^ (is_region_boundary(e) ? Index(1) : Index(0));
        stack.push_back(neighbor);
      }
    }
  }

  template <typename EdgesPolicy, typename IsBoundaryRange,
            typename IntPointsPolicy>
  auto make_arranged_constraints(const tf::edges<EdgesPolicy> &edges,
                                 const IsBoundaryRange &is_boundary,
                                 const tf::points<IntPointsPolicy> &int_pts)
      -> void {
    _constraint_edges.clear();
    _constraint_edges.reserve(edges.size());
    _input_is_boundary.clear();
    _input_is_boundary.reserve(edges.size());

    for (Index i = 0; i < static_cast<Index>(edges.size()); ++i) {
      auto edge = edges[i];
      Index a = Index(edge[0]);
      Index b = Index(edge[1]);
      if (a != b) {
        _constraint_edges.push_back({a, b});
        _input_is_boundary.push_back(is_boundary[i] ? std::uint8_t(1)
                                                    : std::uint8_t(0));
      }
    }

    _final_constraint_edges.clear();
    _final_is_boundary.clear();

    if (_constraint_edges.size() == 0) {
      auto cleaned_pair = tf::cleaned<Index>(int_pts, tf::return_index_map);
      _points = std::move(cleaned_pair.first);
      _index_map = std::move(cleaned_pair.second);
      return;
    }

    _arranged_constraint_edges.clear();
    _arranged_is_boundary.clear();

    {
      auto constraint_segments = tf::make_segments(
          tf::make_edges(tf::make_range(_constraint_edges)), int_pts);

      _constraint_em.build(constraint_segments);

      _constraint_tree.build(constraint_segments, tf::config_tree(4, 4));

      auto tagged = constraint_segments | tf::tag(_constraint_em) |
                    tf::tag(_constraint_tree);

      _constraint_si.build(tagged);

      if (_constraint_si.intersection_points().size() == 0) {
        _arranged_constraint_edges.reserve(_constraint_edges.size());
        _arranged_is_boundary.reserve(_constraint_edges.size());
        for (Index i = 0; i < static_cast<Index>(_constraint_edges.size());
             ++i) {
          _arranged_constraint_edges.push_back(_constraint_edges[i]);
          _arranged_is_boundary.push_back(_input_is_boundary[i]);
        }
      } else {
        _constraint_sig.build(_constraint_si, constraint_segments);

        auto n_original_points = static_cast<Index>(int_pts.size());

        _split_constraint_edge.allocate(_constraint_edges.size());
        tf::parallel_fill(_split_constraint_edge, false);
        for (auto origin : _constraint_sig.origin_edges())
          _split_constraint_edge[origin] = true;

        _arranged_constraint_edges.reserve(
            _constraint_edges.size() + _constraint_sig.flat_sub_edges().size());
        _arranged_is_boundary.reserve(_arranged_constraint_edges.capacity());

        for (Index i = 0; i < static_cast<Index>(_constraint_edges.size());
             ++i) {
          if (!_split_constraint_edge[i]) {
            _arranged_constraint_edges.push_back(_constraint_edges[i]);
            _arranged_is_boundary.push_back(_input_is_boundary[i]);
          }
        }

        auto map_vertex = [&](const auto &v) -> Index {
          if (v.source == tf::intersect::graph::vertex_source::original)
            return Index(v.id);
          return Index(n_original_points + v.id);
        };

        // Sub-edges from each split constraint inherit its is_boundary
        // flag. Walk per-group so we know which origin each sub-edge
        // came from.
        auto sub_groups = _constraint_sig.sub_edges();
        auto origins = _constraint_sig.origin_edges();
        for (Index g = 0; g < static_cast<Index>(origins.size()); ++g) {
          std::uint8_t origin_b = _input_is_boundary[origins[g]];
          for (const auto &edge : sub_groups[g]) {
            Index a = map_vertex(edge[0]);
            Index b = map_vertex(edge[1]);
            if (a != b) {
              _arranged_constraint_edges.push_back({a, b});
              _arranged_is_boundary.push_back(origin_b);
            }
          }
        }
      }
    }

    _augmented_points.clear();
    _augmented_points.reserve(int_pts.size() + _constraint_sig.points().size());
    for (const auto &p : int_pts)
      _augmented_points.push_back(p);
    for (const auto &p : _constraint_sig.points())
      _augmented_points.push_back(p);

    auto cleaned_augmented = tf::cleaned<Index>(
        tf::make_points(_augmented_points), tf::return_index_map);
    _points = std::move(cleaned_augmented.first);
    auto &aug_im = cleaned_augmented.second;

    auto n_input = static_cast<Index>(int_pts.size());
    _index_map = tf::compose_index_maps(
        build_original_to_arrangement_index_map(n_input), aug_im);

    _final_constraint_edges.reserve(_arranged_constraint_edges.size());
    _final_is_boundary.reserve(_arranged_constraint_edges.size());
    for (Index i = 0; i < static_cast<Index>(_arranged_constraint_edges.size());
         ++i) {
      auto &edge = _arranged_constraint_edges[i];
      Index a = aug_im.f()[edge[0]];
      Index b = aug_im.f()[edge[1]];
      if (a != b) {
        _final_constraint_edges.push_back({a, b});
        _final_is_boundary.push_back(_arranged_is_boundary[i]);
      }
    }
  }

  static auto opp(Index e) -> Index { return e ^ Index(1); }

  auto next_e(Index e) const -> Index { return _edges[e].next; }
  auto prev_e(Index e) const -> Index { return _edges[e].prev; }

  auto origin(Index e) const -> Index { return _edges[e].vertex; }
  auto target(Index e) const -> Index { return _edges[opp(e)].vertex; }

  template <typename F> auto for_each_outgoing(Index v, F &&f) const -> void {
    Index first = _v_first_edge[v];
    if (first == k_none)
      return;

    Index e = first;
    do {
      if (!f(e))
        return;

      e = _edges[e].next;
    } while (e != first);
  }

  auto edge_between(Index a, Index b) const -> Index {
    Index found = k_none;

    for_each_outgoing(a, [&](Index e) {
      if (target(e) == b) {
        found = e;
        return false;
      }

      return true;
    });

    return found;
  }

  auto link_into_ring(Index new_e, Index v, Index after) -> void {
    _edges[new_e].vertex = v;

    if (after != k_none) {
      Index before = _edges[after].next;

      _edges[new_e].prev = after;
      _edges[new_e].next = before;

      _edges[after].next = new_e;
      _edges[before].prev = new_e;
    } else {
      _v_first_edge[v] = new_e;

      _edges[new_e].prev = new_e;
      _edges[new_e].next = new_e;
    }
  }

  auto create_edge(Index a, Index b, Index after_a, Index after_b,
                   bool boundary) -> Index {
    Index e_ab = static_cast<Index>(_edges.size());
    Index e_ba = e_ab + Index(1);

    _edges.push_back(half_edge{k_none, k_none, k_none, boundary, false, false});
    _edges.push_back(half_edge{k_none, k_none, k_none, boundary, false, false});

    link_into_ring(e_ab, a, after_a);
    link_into_ring(e_ba, b, after_b);

    return e_ab;
  }

  auto create_edge_reusing(Index a, Index b, Index after_a, Index after_b,
                           Index reused) -> Index {
    Index e_ab = reused;
    Index e_ba = opp(reused);

    _edges[e_ab] = half_edge{k_none, k_none, k_none, false, false, false};
    _edges[e_ba] = half_edge{k_none, k_none, k_none, false, false, false};

    link_into_ring(e_ab, a, after_a);
    link_into_ring(e_ba, b, after_b);

    return e_ab;
  }

  auto mark_initial_boundary_edge(Index e) -> void {
    _edges[e].boundary = true;
    _edges[opp(e)].boundary = true;

    _edges[e].delaunay = true;
    _edges[opp(e)].delaunay = true;
  }

  auto next_boundary(Index e) const -> Index { return next_e(opp(e)); }
  auto prev_boundary(Index e) const -> Index { return opp(prev_e(e)); }

  auto build_delaunay(Index n) -> bool {
    Index e10 = create_edge(Index(1), Index(0), k_none, k_none,
                            /*boundary=*/true);

    mark_initial_boundary_edge(e10);
    _last_edge = e10;

    for (Index i = 2; i < n; ++i)
      insert_vertex(i);

    return true;
  }

  auto insert_vertex(Index p) -> void {
    auto is_visible = [&](Index e) -> bool {
      return orient2d_sign(origin(e), target(e), p) > 0;
    };

    Index last_visible_fwd = _last_edge;
    Index last_visible_bwd = prev_boundary(_last_edge);
    Index prev_visible_bwd = _last_edge;

    while (is_visible(last_visible_fwd))
      last_visible_fwd = next_boundary(last_visible_fwd);

    while (is_visible(last_visible_bwd)) {
      prev_visible_bwd = last_visible_bwd;
      last_visible_bwd = prev_boundary(last_visible_bwd);
    }

    if (last_visible_fwd == prev_visible_bwd) {
      // All points so far are collinear; extend the hull chain only.
      Index prev_v = static_cast<Index>(p - Index(1));

      _last_edge = create_edge(p, prev_v, k_none, _last_edge,
                               /*boundary=*/true);

      mark_initial_boundary_edge(_last_edge);
      return;
    }

    Index current = prev_visible_bwd;
    Index last_added = k_none;

    while (current != last_visible_fwd) {
      Index edge0 = current;

      Index current_vertex = origin(current);
      Index next_vertex = target(current);

      Index next_he = next_boundary(current);

      Index edge1 = last_added != k_none ? last_added
                                         : create_edge(p, current_vertex,
                                                       k_none, prev_e(current),
                                                       /*boundary=*/false);

      Index edge2 = create_edge(p, next_vertex, prev_e(edge1), opp(current),
                                /*boundary=*/false);

      _edges[opp(current)].boundary = false;

      if (last_added == k_none) {
        _edges[edge1].boundary = true;
      }

      add_to_flip_stack(edge0, p);

      last_added = edge2;
      current = next_he;
      ++_n_triangles;
    }

    _edges[opp(last_added)].boundary = true;

    _last_edge = last_added;

    delaunay_flip();
  }

  auto add_to_flip_stack(Index e, Index opposite_vertex) -> void {
    _edges[e].delaunay = false;
    _edges[opp(e)].delaunay = false;

    _flip_stack.push_back(flip_check{e, origin(e), target(e), opposite_vertex});
  }

  auto delaunay_flip() -> void {
    while (_flip_stack.size() > 0) {
      flip_check check = _flip_stack.back();
      _flip_stack.pop_back();

      Index e01 = check.e;
      Index e10 = opp(e01);

      if (origin(e01) != check.v0 || target(e01) != check.v1)
        continue;

      if (_edges[e01].delaunay || _edges[e10].delaunay ||
          _edges[e01].constrained || _edges[e10].constrained)
        continue;

      if (_edges[e01].boundary || _edges[e10].boundary) {
        _edges[e01].delaunay = true;
        _edges[e10].delaunay = true;
        continue;
      }

      Index v0 = origin(e01);
      Index v1 = origin(e10);

      Index other0 = origin(opp(next_e(e01)));
      Index other1 = origin(opp(next_e(e10)));

      if (other0 == other1)
        continue;

      if (tf::exact::incircle(pt(v0), pt(v1), pt(other1), pt(other0)) > 0) {
        Index opposite_vertex = check.opposite_vertex;

        flip_edge(e01);

        if (opposite_vertex == other0) {
          add_to_flip_stack(prev_e(e01), opposite_vertex);
          add_to_flip_stack(next_e(e01), opposite_vertex);
        } else {
          add_to_flip_stack(next_e(e10), opposite_vertex);
          add_to_flip_stack(prev_e(e10), opposite_vertex);
        }
      }

      _edges[e01].delaunay = true;
      _edges[e10].delaunay = true;
    }
  }

  auto flip_edge(Index e_idx) -> void {
    Index e_opp_idx = opp(e_idx);

    Index e01i = _edges[e_idx].prev;
    Index e03i = _edges[e_idx].next;
    Index e21i = _edges[e_opp_idx].next;
    Index e23i = _edges[e_opp_idx].prev;

    _edges[e01i].next = e03i;
    _edges[e03i].prev = e01i;

    _edges[e23i].next = e21i;
    _edges[e21i].prev = e23i;

    _v_first_edge[_edges[e_idx].vertex] = e01i;
    _v_first_edge[_edges[e_opp_idx].vertex] = e23i;

    Index e10i = opp(e01i);
    Index e12i = opp(e21i);
    Index e30i = opp(e03i);
    Index e32i = opp(e23i);

    _edges[e_idx].prev = e12i;
    _edges[e_idx].next = e10i;

    _edges[e10i].prev = e_idx;
    _edges[e12i].next = e_idx;

    _edges[e_opp_idx].prev = e30i;
    _edges[e_opp_idx].next = e32i;

    _edges[e32i].prev = e_opp_idx;
    _edges[e30i].next = e_opp_idx;

    _edges[e_idx].vertex = _edges[e10i].vertex;
    _edges[e_opp_idx].vertex = _edges[e32i].vertex;

    _edges[e_idx].boundary = false;
    _edges[e_opp_idx].boundary = false;
    _edges[e_idx].constrained = false;
    _edges[e_opp_idx].constrained = false;

    _edges[e_idx].delaunay = false;
    _edges[e_opp_idx].delaunay = false;
  }

  auto unlink_edge(Index e) -> void {
    Index eo = opp(e);
    _v_first_edge[origin(e)] = next_e(e);
    _v_first_edge[origin(eo)] = next_e(eo);

    _edges[next_e(e)].prev = prev_e(e);
    _edges[prev_e(e)].next = next_e(e);
    _edges[next_e(eo)].prev = prev_e(eo);
    _edges[prev_e(eo)].next = next_e(eo);
  }

  // Marks both halves of `e` as constrained. The previous constraint
  // state is upgraded but never downgraded, so a non-boundary
  // constraint cannot overwrite a boundary one when two input
  // constraints share the same edge.
  auto mark_constrained(Index e, bool is_boundary) -> void {
    auto kind = is_boundary ? k_boundary_constrained : k_constrained;
    auto upgrade = [&](Index he) {
      if (_edges[he].constrained < kind)
        _edges[he].constrained = kind;
    };
    upgrade(e);
    upgrade(opp(e));
    _edges[e].delaunay = true;
    _edges[opp(e)].delaunay = true;
  }

  auto constrain_single_edge(Index v0, Index v1, bool is_boundary) -> bool {
    Index existing = edge_between(v0, v1);
    if (existing != k_none) {
      mark_constrained(existing, is_boundary);
      return true;
    }

    Index initial = k_none;
    Index vertex_cw = k_none;
    Index vertex_ccw = k_none;
    if (!find_initial_triangle_for_constraint(v0, v1, initial, vertex_cw,
                                              vertex_ccw))
      return false;

    _vertices_cw.clear();
    _vertices_ccw.clear();
    _deleted_edges.clear();

    if (!remove_inner_triangles(v0, v1, vertex_cw, vertex_ccw, initial))
      return false;

    Index before_v0 = edge_between(v0, _vertices_ccw[1]);
    Index before_v1 =
        edge_between(v1, _vertices_cw[_vertices_cw.size() - std::size_t(2)]);

    Index constrained = create_edge_reusing(v0, v1, before_v0, before_v1,
                                            _deleted_edges.back());
    _deleted_edges.pop_back();
    mark_constrained(constrained, is_boundary);

    return retriangulate_constraint_side(_vertices_cw, true) &&
           retriangulate_constraint_side(_vertices_ccw, false);
  }

  auto point_between_on_dominant_axis(Index a, Index b, Index p) const -> bool {
    auto pa = pt(a);
    auto pb = pt(b);
    auto pp = pt(p);
    auto dx = pa[0] > pb[0] ? pa[0] - pb[0] : pb[0] - pa[0];
    auto dy = pa[1] > pb[1] ? pa[1] - pb[1] : pb[1] - pa[1];
    if (dx > dy) {
      auto min_v = pa[0] < pb[0] ? pa[0] : pb[0];
      auto max_v = pa[0] < pb[0] ? pb[0] : pa[0];
      return min_v <= pp[0] && pp[0] <= max_v;
    }
    auto min_v = pa[1] < pb[1] ? pa[1] : pb[1];
    auto max_v = pa[1] < pb[1] ? pb[1] : pa[1];
    return min_v <= pp[1] && pp[1] <= max_v;
  }

  auto find_initial_triangle_for_constraint(Index v0, Index v1, Index &initial,
                                            Index &vertex_cw,
                                            Index &vertex_ccw) const -> bool {
    bool found = false;
    bool ok = true;

    for_each_outgoing(v0, [&](Index e) {
      Index prev_vertex = origin(opp(next_e(e)));
      Index next_vertex = target(e);

      int orient_next = orient2d_sign(v0, v1, next_vertex);
      int orient_prev = orient2d_sign(v0, v1, prev_vertex);

      if (orient_prev == 0 || orient_next == 0) {
        Index collinear = orient_prev == 0 ? prev_vertex : next_vertex;
        if (point_between_on_dominant_axis(v0, v1, collinear)) {
          ok = false;
          found = true;
          return false;
        }
        return true;
      }

      if (orient_prev < 0 && orient_next > 0) {
        initial = e;
        vertex_cw = prev_vertex;
        vertex_ccw = next_vertex;
        found = true;
        return false;
      }

      return true;
    });

    return found && ok;
  }

  auto remove_inner_triangles(Index v0, Index v1, Index vertex_cw,
                              Index vertex_ccw, Index initial) -> bool {
    _vertices_cw.push_back(v0);
    _vertices_cw.push_back(vertex_cw);
    _vertices_ccw.push_back(v0);
    _vertices_ccw.push_back(vertex_ccw);

    Index edge_across = prev_e(opp(initial));

    while (true) {
      if (_edges[edge_across].constrained)
        return false;

      Index edge_of_third = opp(prev_e(edge_across));
      Index third_vertex = origin(edge_of_third);

      Index next_edge_across = k_none;
      bool reached = third_vertex == v1;
      if (!reached) {
        int orientation = orient2d_sign(v0, v1, third_vertex);
        if (orientation == 0)
          return false;

        if (orientation < 0) {
          vertex_cw = third_vertex;
          _vertices_cw.push_back(third_vertex);
          next_edge_across = opp(edge_of_third);
        } else {
          vertex_ccw = third_vertex;
          _vertices_ccw.push_back(third_vertex);
          next_edge_across = prev_e(edge_of_third);
        }
      }

      unlink_edge(edge_across);
      _deleted_edges.push_back(edge_across);

      if (reached)
        break;

      edge_across = next_edge_across;
    }

    _vertices_cw.push_back(v1);
    _vertices_ccw.push_back(v1);
    return _vertices_cw.size() >= 3 && _vertices_ccw.size() >= 3;
  }

  auto take_deleted_edge() -> Index {
    Index e = _deleted_edges.back();
    _deleted_edges.pop_back();
    return e;
  }

  auto retriangulate_constraint_side(const tf::buffer<Index> &vertices,
                                     bool is_cw) -> bool {
    int required_orientation = is_cw ? 1 : -1;

    _retriangulation_stack.clear();
    _retriangulation_stack.push_back(vertices[0]);
    _retriangulation_stack.push_back(vertices[1]);

    for (Index i = 2; i < static_cast<Index>(vertices.size()); ++i) {
      Index current_vertex = vertices[i];

      while (true) {
        Index prev_prev = _retriangulation_stack[_retriangulation_stack.size() -
                                                 std::size_t(2)];
        Index prev = _retriangulation_stack[_retriangulation_stack.size() -
                                            std::size_t(1)];

        if (orient2d_sign(prev_prev, prev, current_vertex) !=
            required_orientation)
          break;

        Index opposite_edge = k_none;
        if (is_cw) {
          if (edge_between(prev_prev, current_vertex) == k_none) {
            opposite_edge = edge_between(prev_prev, prev);
            create_edge_reusing(
                prev_prev, current_vertex, prev_e(opposite_edge),
                edge_between(current_vertex, prev), take_deleted_edge());
          }
        } else {
          if (edge_between(current_vertex, prev_prev) == k_none) {
            opposite_edge = edge_between(prev_prev, prev);
            create_edge_reusing(current_vertex, prev_prev,
                                prev_e(edge_between(current_vertex, prev)),
                                opposite_edge, take_deleted_edge());
          }
        }

        _retriangulation_stack.pop_back();

        if (i == static_cast<Index>(vertices.size()) - Index(1) &&
            opposite_edge == k_none)
          opposite_edge = edge_between(prev_prev, prev);

        if (opposite_edge != k_none)
          add_to_flip_stack(opposite_edge, current_vertex);

        if (_retriangulation_stack.size() < 2)
          break;
      }

      _retriangulation_stack.push_back(current_vertex);
    }

    return _retriangulation_stack.size() == 2;
  }

  auto pt(Index i) const -> tf::point<Int, 2> {
    return tf::make_point(static_cast<Int>(_points[i][0]),
                          static_cast<Int>(_points[i][1]));
  }

  auto orient2d_sign(Index ia, Index ib, Index ic) const -> int {
    auto det = tf::exact::orient2d(pt(ia), pt(ib), pt(ic));

    return (det > 0) ? 1 : (det < 0) ? -1 : 0;
  }

  tf::points_buffer<Int, 2> _points;

  tf::buffer<half_edge> _edges;
  tf::buffer<Index> _v_first_edge;

  tf::buffer<flip_check> _flip_stack;
  tf::buffer<Index> _deleted_edges;
  tf::buffer<Index> _vertices_cw;
  tf::buffer<Index> _vertices_ccw;
  tf::buffer<Index> _retriangulation_stack;
  tf::buffer<std::array<Index, 2>> _constraint_edges;
  tf::buffer<std::array<Index, 2>> _arranged_constraint_edges;
  tf::buffer<std::array<Index, 2>> _final_constraint_edges;
  tf::buffer<std::uint8_t> _input_is_boundary;
  tf::buffer<std::uint8_t> _arranged_is_boundary;
  tf::buffer<std::uint8_t> _final_is_boundary;
  tf::buffer<bool> _split_constraint_edge;
  tf::points_buffer<Int, 2> _augmented_points;
  tf::edge_membership<Index> _constraint_em;
  tf::aabb_tree<Index, Int, 2> _constraint_tree;
  tf::intersections_within_segments<Index, Int, 2, Int> _constraint_si;
  tf::intersect::segment_intersection_graph<Index, 2, Int> _constraint_sig;

  Index _last_edge{k_none};
  Index _n_triangles{0};

  tf::exact_coordinate_converter<Int, Coord, 2> _converter;
  tf::buffer<Index> _region_labels;
  tf::index_map_buffer<Index> _index_map;
};

} // namespace tf
