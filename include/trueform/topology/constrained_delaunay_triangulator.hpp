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
#include "./detail/size_adaptive.hpp"
#include "./topo_type.hpp"
#include <array>
#include <cstdint>
#include <algorithm>

#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

namespace tf {


/// @ingroup topology
/// @brief Constrained Delaunay triangulation of 2D point sets.
///
/// Builds an exact-predicate Delaunay triangulation of the cleaned input
/// points, recovers the constraints, and labels the resulting planar
/// regions. The core is incremental Bowyer-Watson-style insertion in BRIO
/// order (randomized rounds of Hilbert-sorted points) with a remembering
/// point-location walk and apex-routed Lawson flips -- O(N log N) even
/// on structured near-cocircular inputs. Tiny inputs (< 64 points) insert in index
/// order and run every internal primitive serially, so repeated small
/// builds from parallel workers are allocation-free after warm-up.
///
/// Two constraint modes: `split_constraints = true` arranges intersecting
/// constraint edges (intersections may introduce additional output
/// points); `split_constraints = false` preserves constraints verbatim and
/// fails the build if they cross. Repeated boundary marking of one edge
/// TOGGLES its region-separation parity (a zero-width slit crosses the
/// boundary twice, i.e. not at all).
///
/// Input points are converted to `Int`, welded exactly, and exposed
/// through `points()`.
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
    _points.clear();
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
    _insert_order.clear();
    _locate_hint = k_none;
    _n_real = Index(0);
  }

  /// @brief Number of real (non-super) output points. The super-triangle
  /// vertices occupy the three trailing slots of `points()` and are not
  /// referenced by any emitted face.
  auto n_real() const -> Index { return _n_real; }

  /// @brief Output points indexed by `make_faces()`.
  ///
  /// Returned coordinates are deconverted from the internal integer
  /// representation back to `Coord`.
  auto points() const { return tf::make_points(_points); }

  auto converted_points() const {
    return tf::make_points(tf::make_mapped_range(
        _points, [this](const auto &p) { return _converter.deconvert(p); }));
  }

  auto n_triangles() const -> Index { return _n_triangles; }

  /// @brief Visit faces with full adjacency straight off the DCEL:
  /// f(i, v0, v1, v2, label, nbrs, cons): nbrs[k] is the face index
  /// across edge (vk, vk+1) or -1, cons[k] whether that edge is
  /// constrained. Face ids and order match make_faces/region_labels.
  /// Valid after build; assumes no super vertices (dual-path build).
  template <typename F> auto for_each_face_adjacency(F &&f) const -> void {
    auto nbr = [&](Index e) -> Index {
      return _scratch_tri_of_edge[std::size_t(opp(e))];
    };
    auto con = [&](Index e) -> bool { return _edges[e].constrained != 0; };
    for (Index i = 0; i < Index(_scratch_first_edge.size()); ++i) {
      Index e0 = _scratch_first_edge[std::size_t(i)];
      Index e1 = prev_e(opp(e0));
      Index e2 = prev_e(opp(e1));
      f(i, _edges[e0].vertex, _edges[e1].vertex, _edges[e2].vertex,
        _region_labels[std::size_t(i)],
        std::array<Index, 3>{nbr(e0), nbr(e1), nbr(e2)},
        std::array<bool, 3>{con(e0), con(e1), con(e2)});
    }
  }

  /// @brief Build and return triangles of the constrained triangulation.
  auto make_faces() const -> tf::blocked_buffer<Index, 3> {
    // With no region-boundary constraints the labels are all zero, so face
    // order carries no meaning and extraction can go wide (order-free).
    if (!has_region_boundary_constraints())
      return make_faces_parallel();

    tf::blocked_buffer<Index, 3> out;
    out.reserve(static_cast<std::size_t>(_n_triangles));

    auto n_he = static_cast<Index>(_edges.size());

    // an ascending scan meets each face first through its minimum
    // half-edge id, so emitting there needs no visited marks
    for (Index e0 = 0; e0 < n_he; ++e0) {
      if (_edges[e0].boundary)
        continue;

      Index e1 = prev_e(opp(e0));
      Index e2 = prev_e(opp(e1));

      if (e1 == k_none || e2 == k_none)
        continue;

      if (_edges[e1].boundary || _edges[e2].boundary)
        continue;

      if (e1 < e0 || e2 < e0)
        continue;

      Index v0 = _edges[e0].vertex;
      Index v1 = _edges[e1].vertex;
      Index v2 = _edges[e2].vertex;
      if (v0 >= _n_real || v1 >= _n_real || v2 >= _n_real)
        continue; // triangle incident to a super vertex — outside the real hull

      out.push_back({v0, v1, v2});
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

  /// Split-constraints builds only: invoke `f(point, parents)` for
  /// every arrangement point lying strictly inside a constraint —
  /// `point` addresses points(), `parents` is the range of input
  /// constraints the point is interior to (one entry = a T-junction at
  /// an existing input vertex; duplicated constraints all appear).
  template <typename F>
  auto for_each_constraint_crossing(F &&f) const -> void {
    tf::buffer<std::array<Index, 2>> per_point;
    for (auto group : _constraint_si.intersections())
      for (const auto &rec : group) {
        if (rec.target.label != tf::topo_type::edge)
          continue;
        per_point.push_back(
            {_constraint_sig.point_remap()[rec.id], Index(rec.object)});
      }
    std::sort(per_point.begin(), per_point.end());
    const Index n_input = static_cast<Index>(_index_map.f().size());
    tf::buffer<Index> parents;
    for (std::size_t i = 0; i < per_point.size();) {
      std::size_t j = i + 1;
      while (j < per_point.size() && per_point[j][0] == per_point[i][0])
        ++j;
      parents.clear();
      for (std::size_t k = i; k < j; ++k)
        parents.push_back(per_point[k][1]);
      f(n_input + per_point[i][0], tf::make_range(parents));
      i = j;
    }
  }

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
             const IsBoundaryRange &is_boundary,
             bool split_constraints = true) -> bool {
    clear();
    _converter = tf::make_exact_coordinate_converter<Int, Coord>(pts);

    if (split_constraints) {
      auto int_pts = tf::make_points(tf::make_mapped_range(
          pts, [this](const auto &p) { return _converter(p); }));

      make_arranged_constraints(edges, is_boundary, int_pts);
    } else if (!make_preserved_constraints(pts, edges, is_boundary)) {
      return false;
    }

    auto n = static_cast<Index>(_points.size());
    _n_real = n; // no super vertices in the dual-path build
    if (n < 3)
      return true;

    _v_first_edge.allocate(n);
    topology::detail::fill_auto(_v_first_edge, k_none);
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
             const tf::edges<EdgesPolicy> &edges,
             bool split_constraints = true) -> bool {
    return build(pts, edges, tf::make_constant_range(true, edges.size()),
                 split_constraints);
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

  auto has_region_boundary_constraints() const -> bool {
    for (auto b : _final_is_boundary)
      if (b)
        return true;
    return false;
  }

  // Order-free triangle extraction: a face is emitted from its minimum
  // half-edge. Chunked count + prefix + write, no per-face push_back.
  auto make_faces_parallel() const -> tf::blocked_buffer<Index, 3> {
    tf::blocked_buffer<Index, 3> out;
    auto n_he = static_cast<Index>(_edges.size());
    if (n_he == 0)
      return out;

    auto canonical = [this](Index e0) -> bool {
      if (_edges[e0].boundary)
        return false;
      Index e1 = prev_e(opp(e0));
      Index e2 = prev_e(opp(e1));
      if (e1 == k_none || e2 == k_none || e1 < e0 || e2 < e0)
        return false;
      if (_edges[e1].boundary || _edges[e2].boundary)
        return false;
      return _edges[e0].vertex < _n_real && _edges[e1].vertex < _n_real &&
             _edges[e2].vertex < _n_real;
    };

    const Index chunk = 1 << 14;
    Index n_chunks = (n_he + chunk - 1) / chunk;
    tf::buffer<Index> counts;
    counts.allocate(static_cast<std::size_t>(n_chunks));
    tbb::parallel_for(
        tbb::blocked_range<Index>(Index(0), n_chunks),
        [&](const tbb::blocked_range<Index> &r) {
          for (Index c = r.begin(); c != r.end(); ++c) {
            Index end = std::min<Index>(n_he, (c + 1) * chunk);
            Index count = 0;
            for (Index e = c * chunk; e < end; ++e)
              count += canonical(e) ? Index(1) : Index(0);
            counts[c] = count;
          }
        });

    Index total = 0;
    for (Index c = 0; c < n_chunks; ++c) {
      Index t = counts[c];
      counts[c] = total;
      total += t;
    }

    out.allocate(static_cast<std::size_t>(total));
    Index *base = out.data_buffer().data();
    tbb::parallel_for(
        tbb::blocked_range<Index>(Index(0), n_chunks),
        [&](const tbb::blocked_range<Index> &r) {
          for (Index c = r.begin(); c != r.end(); ++c) {
            Index end = std::min<Index>(n_he, (c + 1) * chunk);
            Index *w = base + std::size_t(3) * counts[c];
            for (Index e0 = c * chunk; e0 < end; ++e0) {
              if (!canonical(e0))
                continue;
              Index e1 = prev_e(opp(e0));
              Index e2 = prev_e(opp(e1));
              *w++ = _edges[e0].vertex;
              *w++ = _edges[e1].vertex;
              *w++ = _edges[e2].vertex;
            }
          }
        });

    return out;
  }

  auto compute_region_labels() -> void {
    _region_labels.clear();
    auto n_he = static_cast<Index>(_edges.size());
    if (n_he == 0)
      return;

    auto &tri_of_edge = _scratch_tri_of_edge;
    tri_of_edge.clear();
    tri_of_edge.allocate(n_he);
    topology::detail::fill_auto(tri_of_edge, k_none);

    auto &first_edge_of_tri = _scratch_first_edge;
    first_edge_of_tri.clear();
    first_edge_of_tri.reserve(static_cast<std::size_t>(_n_triangles));

    for (Index e0 = 0; e0 < n_he; ++e0) {
      if (_edges[e0].boundary)
        continue;

      Index e1 = prev_e(opp(e0));
      Index e2 = prev_e(opp(e1));
      if (e1 == k_none || e2 == k_none || _edges[e1].boundary ||
          _edges[e2].boundary)
        continue;
      if (e1 < e0 || e2 < e0)
        continue;

      Index tri = static_cast<Index>(first_edge_of_tri.size());
      first_edge_of_tri.push_back(e0);
      tri_of_edge[e0] = tri;
      tri_of_edge[e1] = tri;
      tri_of_edge[e2] = tri;
    }

    if (first_edge_of_tri.size() == 0)
      return;

    // Parity flood over ALL triangles (real + super). The super triangles are
    // the "outside" region: seeding from a super-triangle boundary edge gives
    // them parity 0, and crossing a region-boundary constraint toggles parity.
    auto &parity = _scratch_parity;
    parity.clear();
    parity.allocate(first_edge_of_tri.size());
    topology::detail::fill_auto(parity, k_none);

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

    auto &stack = _scratch_stack;
    stack.clear();
    stack.push_back(start_tri);
    parity[start_tri] =
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
        if (neighbor == k_none || parity[neighbor] != k_none)
          continue;

        parity[neighbor] =
            parity[tri] ^ (is_region_boundary(e) ? Index(1) : Index(0));
        stack.push_back(neighbor);
      }
    }

    // Emit labels for the REAL triangles only, in the exact order
    // `make_faces` visits them (both are the ascending min-edge-id scan),
    // so `region_labels()[i]` matches face `i`.
    _region_labels.reserve(first_edge_of_tri.size());
    for (Index t = 0; t < static_cast<Index>(first_edge_of_tri.size());
         ++t) {
      Index e0 = first_edge_of_tri[std::size_t(t)];
      Index e1 = prev_e(opp(e0));
      Index e2 = prev_e(opp(e1));
      if (origin(e0) >= _n_real || origin(e1) >= _n_real ||
          origin(e2) >= _n_real)
        continue;
      _region_labels.push_back(parity[std::size_t(t)]);
    }
  }

  // Preserved-constraints preparation: lex-sort + exact-weld the converted
  // points and pass the constraints through verbatim. Fills the same state
  // make_arranged_constraints does.
  template <typename PointsPolicy, typename EdgesPolicy,
            typename IsBoundaryRange>
  auto make_preserved_constraints(const tf::points<PointsPolicy> &pts,
                                  const tf::edges<EdgesPolicy> &edges,
                                  const IsBoundaryRange &is_boundary) -> bool {
    auto n_input = static_cast<Index>(pts.size());
    auto &converted = _scratch_pts;
    converted.clear();
    converted.allocate(n_input);
    if (std::size_t(n_input) < topology::detail::k_serial_cutoff)
      for (Index i = 0; i < n_input; ++i)
        converted.points()[i] = _converter(pts[i]);
    else
      tf::parallel_for_each(tf::make_sequence_range(n_input), [&](Index i) {
        converted.points()[i] = _converter(pts[i]);
      });

    auto &order = _scratch_order;
    order.clear();
    order.allocate(n_input);
    topology::detail::iota_auto(order, Index(0));
    topology::detail::sort_auto(order.begin(), order.end(), [&](Index a, Index b) {
      const auto &pa = converted.points()[a];
      const auto &pb = converted.points()[b];
      return pa[0] < pb[0] || (pa[0] == pb[0] && pa[1] < pb[1]);
    });

    _index_map.f().allocate(n_input);
    for (Index i = 0; i < n_input; ++i) {
      auto p = converted.points()[order[i]];
      if (i == 0 || p[0] != _points.points()[_points.size() - 1][0] ||
          p[1] != _points.points()[_points.size() - 1][1]) {
        _index_map.kept_ids().push_back(order[i]);
        _points.push_back(p);
      }
      _index_map.f()[order[i]] = Index(_points.size() - 1);
    }

    auto n_edges = static_cast<Index>(edges.size());
    _final_constraint_edges.reserve(n_edges);
    _final_is_boundary.reserve(n_edges);
    auto ib = is_boundary.begin();
    for (Index i = 0; i < n_edges; ++i, ++ib) {
      Index a = _index_map.f()[edges[i][0]];
      Index b = _index_map.f()[edges[i][1]];
      if (a == b)
        return false; // degenerate constraint: coincident endpoints
      _final_constraint_edges.push_back({a, b});
      _final_is_boundary.push_back(*ib ? 1 : 0);
    }
    return true;
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

  // Incremental Delaunay over the real points (no super vertices). BRIO
  // insertion keeps point location and the flip cascade local, so flip
  // counts stay O(1) amortized even on near-cocircular inputs.
  //
  // Phase A: insert in index order until the first non-degenerate triangle
  // exists (a collinear prefix has no face to walk from).
  //
  // Phase B: insert the rest in Hilbert order via point-location walk +
  // interior split; points outside the current hull reuse the hull-fan
  // `insert_vertex`, seeded at the boundary edge `locate` returns so the fan
  // walk is local (O(1) amortized) rather than a full hull traversal.
  auto build_delaunay(Index n) -> bool {
    Index e10 = create_edge(Index(1), Index(0), k_none, k_none,
                            /*boundary=*/true);
    mark_initial_boundary_edge(e10);
    _last_edge = e10;

    Index i = 2;
    for (; i < n && _n_triangles == 0; ++i)
      insert_vertex(i);

    if (i >= n)
      return true;

    build_morton_order(i, n);
    _locate_hint = find_any_interior_edge();

    for (Index k = 0; k < static_cast<Index>(_insert_order.size()); ++k) {
      insert_incremental(_insert_order[k]);
    }

    return true;
  }

  // Hilbert-curve index of a point on a 2^k x 2^k grid (k = k_order_bits).
  // Hilbert order has far better locality than Morton (Z-order) — consecutive
  // points stay spatially adjacent even for curve-like inputs, which keeps the
  // per-insertion flip cascade local (O(1) amortized instead of super-linear).
  static constexpr std::uint32_t k_order_bits = 21;
  static auto hilbert_code(std::uint32_t x, std::uint32_t y) -> std::uint64_t {
    std::uint64_t d = 0;
    for (std::uint32_t s = k_order_bits; s-- > 0;) {
      std::uint32_t rx = (x >> s) & 1u;
      std::uint32_t ry = (y >> s) & 1u;
      d += static_cast<std::uint64_t>((3u * rx) ^ ry) << (2u * s);
      // Rotate the quadrant so the curve stays continuous.
      if (ry == 0u) {
        if (rx == 1u) {
          std::uint32_t m = (1u << k_order_bits) - 1u;
          x = m - x;
          y = m - y;
        }
        std::uint32_t t = x;
        x = y;
        y = t;
      }
    }
    return d;
  }

  // Fills `_insert_order` with the point indices [begin, end), sorted by the
  // Morton code of their (shifted, range-reduced) coordinates.
  auto build_morton_order(Index begin, Index end) -> void {
    auto min_x = _points[begin][0];
    auto min_y = _points[begin][1];
    auto max_x = min_x;
    auto max_y = min_y;
    for (Index i = begin; i < end; ++i) {
      auto p = _points[i];
      min_x = p[0] < min_x ? p[0] : min_x;
      min_y = p[1] < min_y ? p[1] : min_y;
      max_x = p[0] > max_x ? p[0] : max_x;
      max_y = p[1] > max_y ? p[1] : max_y;
    }
    // Reduce each axis to 21 bits so the interleave fits in 64 bits. Widen to
    // 64-bit before subtracting: converter-range int coords span the full
    // int32 range, so an int32 difference would overflow.
    auto span_x = static_cast<std::uint64_t>(static_cast<std::int64_t>(max_x) -
                                             static_cast<std::int64_t>(min_x));
    auto span_y = static_cast<std::uint64_t>(static_cast<std::int64_t>(max_y) -
                                             static_cast<std::int64_t>(min_y));
    auto span = span_x > span_y ? span_x : span_y;
    std::uint32_t shift = 0;
    while ((span >> shift) > 0x1fffffULL)
      ++shift;

    Index n = end - begin;
    _insert_order.allocate(static_cast<std::size_t>(n));
    // tiny inputs: any insertion order is O(1)-local for the remembering
    // walk; the shuffle + Hilbert coding would cost more than they save
    if (n < Index(64)) {
      for (Index k = 0; k < n; ++k)
        _insert_order[static_cast<std::size_t>(k)] = begin + k;
      return;
    }
    auto &keys = _scratch_keys;
    keys.clear();
    keys.allocate(static_cast<std::size_t>(n));
    auto fill_keys = [&](Index k_lo, Index k_hi) {
      for (Index k = k_lo; k != k_hi; ++k) {
        auto p = _points[begin + k];
        auto x = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(static_cast<std::int64_t>(p[0]) -
                                       static_cast<std::int64_t>(min_x)) >>
            shift);
        auto y = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(static_cast<std::int64_t>(p[1]) -
                                       static_cast<std::int64_t>(min_y)) >>
            shift);
        _insert_order[static_cast<std::size_t>(k)] = begin + k;
        keys[static_cast<std::size_t>(k)] = hilbert_code(x, y);
      }
    };
    if (std::size_t(n) < topology::detail::k_serial_cutoff)
      fill_keys(Index(0), n);
    else
      tbb::parallel_for(tbb::blocked_range<Index>(Index(0), n),
                        [&](const tbb::blocked_range<Index> &r) {
                          fill_keys(r.begin(), r.end());
                        });

    // Biased Randomized Insertion Order (BRIO), matching CGAL's spatial_sort:
    // a random shuffle, then a multiscale recursion that Hilbert-sorts rounds
    // of geometrically increasing size (ratio 0.25). This inserts points in
    // randomized rounds of increasing density — each round well spread over the
    // domain — which makes incremental Delaunay O(N log N). Pure Hilbert order
    // alone is super-linear on structured inputs like a convex curve.
    Index *ord = &_insert_order[0];
    const std::uint64_t *key = &keys[0];

    std::uint64_t s =
        0x9e3779b97f4a7c15ULL ^ (static_cast<std::uint64_t>(n) + 1u) *
                                    0xff51afd7ed558ccdULL;
    auto next_rand = [&]() {
      s ^= s >> 33;
      s *= 0xff51afd7ed558ccdULL;
      s ^= s >> 33;
      return s;
    };
    for (Index k = n - 1; k > 0; --k) {
      Index j = static_cast<Index>(next_rand() % static_cast<std::uint64_t>(k + 1));
      std::swap(ord[k], ord[j]);
    }

    brio_sort(ord, key, begin, Index(0), n);
  }

  // Multiscale Hilbert sort (the BRIO recursion). Keeps the last 25% of the
  // range as its own Hilbert-sorted round, recursing on the first 75%.
  auto brio_sort(Index *ord, const std::uint64_t *key, Index begin, Index lo,
                 Index hi) const -> void {
    auto cmp = [&](Index a, Index b) {
      // total order: tie-break equal Hilbert keys by index, so parallel
      // sort is deterministic (timing-independent) on duplicate cells
      auto ka = key[a - begin], kb = key[b - begin];
      return ka != kb ? ka < kb : a < b;
    };
    Index size = hi - lo;
    Index mid = lo;
    if (size >= 16) {
      mid = lo + (size * Index(3)) / Index(4);
      brio_sort(ord, key, begin, lo, mid);
    }
    // Hilbert-sort this round [mid, hi). Rounds are disjoint, so each sort is
    // independent; the largest rounds dominate and go parallel.
    if (hi - mid > Index(4096))
      tbb::parallel_sort(ord + mid, ord + hi, cmp);
    else
      std::sort(ord + mid, ord + hi, cmp);
  }


  auto find_any_interior_edge() const -> Index {
    auto n_he = static_cast<Index>(_edges.size());
    for (Index e = 0; e < n_he; ++e)
      if (!_edges[e].boundary)
        return e;
    return k_none;
  }

  enum class locate_kind { interior, boundary_edge, exterior };
  struct locate_result {
    locate_kind kind;
    Index he;
  };

  // Walks the triangulation from the current hint to the face containing `p`.
  //   interior      — `he` is a half-edge of the containing face. `p` is
  //                    strictly inside, or on an interior edge (handled by the
  //                    split: the degenerate triangle is removed by the flip).
  //   boundary_edge — `p` lies on the hull edge `he` (a boundary half-edge).
  //   exterior      — `p` is outside the hull, beyond boundary half-edge `he`.
  // Remembering walk. The face cycle (e0, prev(opp(e0)), ...) is CW, so `p`
  // lies in the face of `e0` iff it is not strictly left of any face edge;
  // after crossing an edge, `p` is on the entered edge's right by
  // construction, so each subsequent face costs at most two orientation tests
  // instead of re-orienting the face.
  auto locate(Index p) -> locate_result {
    Index e0 = _locate_hint;
    if (e0 == k_none || _edges[e0].boundary)
      e0 = find_any_interior_edge();

    int s0 = orient2d_sign(origin(e0), target(e0), p);
    if (s0 > 0) {
      if (_edges[opp(e0)].boundary)
        return {locate_kind::exterior, opp(e0)};
      e0 = opp(e0);
      s0 = -1;
    }

    auto n_he = static_cast<Index>(_edges.size());
    for (Index guard = 0; guard <= n_he; ++guard) {
      Index e1 = prev_e(opp(e0));
      Index e2 = prev_e(opp(e1));

      // Fixed (e1, e2) preference: decision-for-decision identical to the
      // reference walk, whose e0 test only ever fires on the first face.
      Index fa = e1;
      Index fb = e2;

      int sa = orient2d_sign(origin(fa), target(fa), p);
      if (sa > 0) {
        if (_edges[opp(fa)].boundary)
          return {locate_kind::exterior, opp(fa)};
        e0 = opp(fa);
        s0 = -1;
        continue;
      }

      int sb = orient2d_sign(origin(fb), target(fb), p);
      if (sb > 0) {
        if (_edges[opp(fb)].boundary)
          return {locate_kind::exterior, opp(fb)};
        e0 = opp(fb);
        s0 = -1;
        continue;
      }

      if (s0 == 0 || sa == 0 || sb == 0) {
        Index fe[3] = {e0, fa, fb};
        int fs[3] = {s0, sa, sb};
        for (int k = 0; k < 3; ++k) {
          if (fs[k] == 0 && point_between_on_dominant_axis(origin(fe[k]),
                                                           target(fe[k]), p)) {
            if (_edges[opp(fe[k])].boundary)
              return {locate_kind::boundary_edge, opp(fe[k])};
            // interior on-edge: the interior split below degenerates one
            // triangle which the Delaunay flip immediately repairs.
            return {locate_kind::interior, e0};
          }
        }
      }

      return {locate_kind::interior, e0};
    }

    return {locate_kind::interior, e0};
  }

  auto insert_incremental(Index p) -> void {
    locate_result loc = locate(p);
    switch (loc.kind) {
    case locate_kind::interior:
      split_triangle(p, loc.he);
      break;
    case locate_kind::boundary_edge:
      split_boundary_edge(p, loc.he);
      break;
    case locate_kind::exterior:
      // Seed the hull-fan walk so it starts on a half-edge visible from p.
      // insert_vertex's _last_edge is an interior half-edge whose opp is the
      // boundary edge and which satisfies is_visible(e) > 0; that is exactly
      // the interior twin of the boundary edge p crossed (loc.he). Seeding
      // with the boundary half-edge itself has the wrong sign and would drop
      // into the collinear branch.
      _last_edge = opp(loc.he);
      insert_vertex(p);
      // Propagate the location hint to an edge incident to the just-inserted
      // point, so the next (Hilbert-adjacent) point's locate walk starts
      // nearby — O(1) amortized instead of walking from a stale hint. Without
      // this, an all-exterior input (a convex curve) degrades to O(N) walks.
      _locate_hint = _last_edge;
      break;
    }
  }

  // Inserts `p` strictly inside the face whose first half-edge is `f0`,
  // splitting one triangle into three. Reuses the existing flip machinery to
  // restore the Delaunay property afterwards.
  //
  // Face half-edges f0:A->B, f1:B->C, f2:C->A (origins A,B,C). The three new
  // spokes are p->A, p->B, p->C; the three original face edges become the
  // outer edges opposite p and are pushed onto the flip stack.
  auto split_triangle(Index p, Index f0) -> void {
    Index f1 = prev_e(opp(f0));
    Index f2 = prev_e(opp(f1));

    Index A = origin(f0);
    Index B = origin(f1);
    Index C = origin(f2);

    // p-ring order p->A, p->B, p->C chained via after_p; the per-vertex
    // anchor is the outgoing face edge (next_e(f_{i+1}) == opp(f_i)).
    Index ea = create_edge(p, A, k_none, f0, /*boundary=*/false);
    Index eb = create_edge(p, B, ea, f1, /*boundary=*/false);
    create_edge(p, C, eb, f2, /*boundary=*/false);

    add_to_flip_stack(f0, p);
    add_to_flip_stack(f1, p);
    add_to_flip_stack(f2, p);

    _n_triangles += Index(2);
    _locate_hint = ea;

    delaunay_flip();
  }

  // Inserts `p`, which lies on the hull boundary half-edge `eb` (u->v), into
  // the single adjacent interior triangle (v,u,w). Splits the boundary edge
  // u-v into u-p and p-v (both boundary) and adds the spoke p-w. The two outer
  // edges u-w and w-v are pushed onto the flip stack.
  auto split_boundary_edge(Index p, Index eb) -> void {
    Index gi = opp(eb);
    Index u = origin(eb);
    Index v = origin(gi);

    Index n1 = prev_e(eb);      // u->w
    Index n2 = prev_e(opp(n1)); // w->v
    Index w = target(n1);

    unlink_edge(eb);

    // p's ring must be ordered p->u, p->w, p->v (so next_e(p->u)=p->w,
    // next_e(p->w)=p->v), which is required by the face_next bookkeeping of
    // the two new triangles (p,u,w) and (v,p,w). Create the edges in that
    // order, chaining each new spoke after the previous one in p's ring.

    // u-p edge reuses the freed (eb,gi) slot: u->p boundary, p->u interior;
    // p->u starts p's ring, u->p inserted after n1=u->w in u's ring.
    Index up = create_edge_reusing(u, p, n1, k_none, eb);
    Index pu = opp(up);
    _edges[up].boundary = true;
    _edges[up].delaunay = true;

    // p-w spoke: interior. p->w after p->u in p's ring; w->p inserted just
    // before opp(n1)=w->u in w's ring, i.e. after n2.
    Index pw = create_edge(p, w, pu, n2, /*boundary=*/false);

    // p-v edge: p->v boundary, v->p interior. p->v after p->w in p's ring;
    // v->p inserted just before opp(n2)=v->w in v's ring.
    Index pv = create_edge(p, v, pw, prev_e(opp(n2)), /*boundary=*/false);
    _edges[pv].boundary = true;
    _edges[pv].delaunay = true;

    add_to_flip_stack(n1, p);
    add_to_flip_stack(n2, p);

    _last_edge = up;
    _locate_hint = pw;
    _n_triangles += Index(1);

    delaunay_flip();
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

      if (in_circle_sign(v0, v1, other1, other0) > 0) {
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
    // repeated boundary marking TOGGLES region separation: a zero-width
    // slit crosses the boundary twice, i.e. not at all
    auto apply = [&](Index he) {
      auto &c = _edges[he].constrained;
      if (is_boundary)
        c = (c == k_boundary_constrained) ? k_constrained
                                          : k_boundary_constrained;
      else if (c == k_unconstrained)
        c = k_constrained;
    };
    apply(e);
    apply(opp(e));
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

  // Filtered orient2d, same shape as the filtered in_circle_sign: double
  // determinant with a Shewchuk error bound, exact int fallback, and the
  // diff-fits-in-2^53 guard for large (int64) coordinates. Uses the exact
  // formula (b-a)x(c-a) so the float and exact paths share a sign convention.
  auto orient2d_sign(Index ia, Index ib, Index ic) const -> int {
    return tf::exact::orient2d_sign(pt(ia), pt(ib), pt(ic));
  }

  auto is_super(Index v) const -> bool { return v >= _n_real; }

  // In-circle test (> 0 == d strictly inside circumcircle of CCW a,b,c) with
  // the super vertices treated as a single point at infinity:
  //   - an infinite d is never inside a finite circle;
  //   - a circle through one infinite + two finite points degenerates to the
  //     line through the two finite points, so the test reduces to the
  //     orientation of those two (kept in cyclic order) and d.
  // Concrete super coordinates are still used for orientation/location, but
  // never for the empty-circle test — for near-collinear integer points the
  // circumradius grows like span^3, so no finite coordinate could sit outside
  // it, which is exactly why the infinite vertex must be symbolic here.
  // Filtered in-circle: a double-precision determinant with a Shewchuk
  // semi-static error bound, falling back to the exact int128 predicate only
  // when the float result is too close to zero to trust. Exact-equivalent.
  //
  // The filter is only applied when the coordinate differences are exactly
  // representable in a double (|diff| < 2^53). For int32 grids that always
  // holds (diffs ~2^32); for int64 grids it may not, and we then go straight
  // to the exact predicate — so large scaled coordinates never give a wrong
  // sign.
  auto in_circle_sign(Index a, Index b, Index c, Index d) const -> int {
    return tf::exact::incircle_sign(pt(a), pt(b), pt(c), pt(d));
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

  // per-build scratch: reused across builds, never freed
  tf::buffer<std::uint64_t> _scratch_keys;
  tf::points_buffer<Int, 2> _scratch_pts;
  tf::buffer<Index> _scratch_order;
  tf::buffer<Index> _scratch_tri_of_edge;
  tf::buffer<Index> _scratch_first_edge;
  tf::buffer<Index> _scratch_parity;
  tf::buffer<Index> _scratch_stack;

  tf::buffer<Index> _insert_order;
  Index _locate_hint{k_none};

  Index _n_real{0};
  Coord _bbox_center_x{0};
  Coord _bbox_center_y{0};
  Coord _bbox_half{0};

};

} // namespace tf
