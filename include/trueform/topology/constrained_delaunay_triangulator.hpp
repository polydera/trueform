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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/edges.hpp"
#include "../core/index_map.hpp"
#include "../core/point.hpp"
#include "../core/points.hpp"
#include "../core/points_buffer.hpp"
#include "../core/range.hpp"
#include "../core/views/constant.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../exact/dyadic_blend.hpp"
#include "../exact/dyadic_ratio.hpp"
#include "../exact/incircle.hpp"
#include "../exact/is_between_on_segment.hpp"
#include "../exact/int32.hpp"
#include "../exact/orient2d.hpp"
#include "../exact/rebase_parameter.hpp"
#include "../exact_coordinate_converter.hpp"
#include "./detail/insertion_order.hpp"
#include <array>
#include <cstddef>
#include <cstdint>


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
/// Two constraint modes: `split_constraints = true` resolves crossings
/// against the live triangulation as they are met, so recovery may
/// introduce output points no input carried; `split_constraints = false`
/// preserves constraints verbatim and fails the build if they cross. Repeated boundary marking of one edge
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
public:
  /// The crossing parameter is the geometry here, so it is carried at the
  /// exact substrate's own width rather than squeezed into Index. At 30
  /// bits a crossing near the end of a long constraint quantises to a
  /// displacement of 1/2^30 of its length — millions of lattice units at
  /// pipeline scale. The width is chosen so that a parameter, and the
  /// products the blend and the rebase form from it, stay inside T1/T2.
  using param_t = typename tf::exact::meta<Int>::param_type;
  static constexpr int k_crossing_param_bits = tf::exact::meta<Int>::param_bits;
  static constexpr int k_max_resolution_depth = 64;

  /// A crossing the incremental resolution created: the point it inserted
  /// and, per crossed constraint, its input id and the parameter of the
  /// crossing along that constraint's whole physical edge.
  struct constraint_owner_t {
    Index input_id;
    param_t t0;
    param_t t1;
  };

  struct incremental_crossing {
    Index point;
    Index id_a;
    Index id_b;
    param_t t_a;
    param_t t_b;
  };

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
    _scratch_first_edge.clear();
    _scratch_tri_of_edge.clear();
    _scratch_parity.clear();
    _points.clear();
    _edges.clear();
    _v_first_edge.clear();
    _flip_stack.clear();
    _deleted_edges.clear();
    _vertices_cw.clear();
    _vertices_ccw.clear();
    _retriangulation_stack.clear();
    _constraint_input_ids.clear();
    _he_constraint.clear();
    _incremental_crossings.clear();
    _final_constraint_edges.clear();
    _final_is_boundary.clear();
    _last_edge = k_none;
    _n_triangles = Index(0);
    _region_labels.clear();
    _index_map.f().clear();
    _index_map.kept_ids().clear();
    _insert_order.clear();
    _locate_hint = k_none;
  }

  /// @brief Number of real (non-super) output points. The super-triangle
  /// vertices occupy the three trailing slots of `points()` and are not
  /// referenced by any emitted face.

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
  /// Valid after build.
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
  /// The triangles, in the order `region_labels()` labels them.
  ///
  /// Reads the canonical half-edge per triangle that `compute_region_labels`
  /// materialised. Deriving the set a second time here is what let the two
  /// disagree: a filter dropped triangles from one and not the other, and a
  /// build reported faces it had no labels for.
  auto make_faces() const -> tf::blocked_buffer<Index, 3> {
    tf::blocked_buffer<Index, 3> out;
    out.allocate(_scratch_first_edge.size());
    tf::parallel_for_each(
        tf::make_sequence_range(Index(_scratch_first_edge.size())),
        [&](Index tri) {
          const Index e0 = _scratch_first_edge[std::size_t(tri)];
          const Index e1 = prev_e(opp(e0));
          const Index e2 = prev_e(opp(e1));
          auto block = out[std::size_t(tri)];
          block[0] = _edges[e0].vertex;
          block[1] = _edges[e1].vertex;
          block[2] = _edges[e2].vertex;
        },
        tf::checked);
    return out;
  }

  /// @brief Index map between original input point space and the
  /// cleaned output point space.
  ///
  /// `f()[input_id]` is the output index of input point `input_id`.
  /// `kept_ids()[output_id]` is the original input ID that survived the
  /// weld for output slot `output_id`, or the sentinel `f().size()` for a
  /// vertex recovery created.
  auto index_map() const -> const tf::index_map_buffer<Index> & {
    return _index_map;
  }

  /// @brief Non-const accessor to allow moving the index map out.
  auto index_map() -> tf::index_map_buffer<Index> & { return _index_map; }

  /// @brief Crossings recovery had to create, as {point, constraint_a,
  ///        constraint_b, t_a, t_b} where each t is a dyadic parameter on
  ///        2^`crossing_param_bits()` along that constraint. The parameter,
  ///        not the coordinate, is the transportable fact: a carrier holding
  ///        the same edge places the split from it exactly.
  auto parameterized_crossings() const
      -> const tf::buffer<incremental_crossing> & {
    return _incremental_crossings;
  }


  static constexpr int crossing_param_bits() { return k_crossing_param_bits; }

  /// @brief Region labels separated by constrained edges, indexed by
  /// triangle (matches the triangle order produced by `make_faces()`).
  ///
  /// Labels are parity values propagated across triangles. Crossing a
  /// constrained edge toggles the label. Computed during `build()`.
  auto region_labels() const { return tf::make_range(_region_labels); }

  /// Split-constraints builds only: invoke `f(point, parents)` for
  /// every point lying strictly inside a constraint —
  /// `point` addresses points(), `parents` is the range of input
  /// constraints the point is interior to (one entry = a T-junction at
  /// an existing input vertex; duplicated constraints all appear).
  template <typename F>
  auto for_each_constraint_crossing(F &&f) const -> void {
    for (const auto &record : _incremental_crossings) {
      // One parent means a T-junction at a vertex that already exists —
      // the contract this function already documents.
      const std::array<Index, 2> parents{record.id_a, record.id_b};
      const std::size_t n = record.id_b == k_none ? 1u : 2u;
      f(record.point, tf::make_range(parents.data(), parents.data() + n));
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
  template <typename PointsPolicy, typename EdgesPolicy, typename Iterator,
            std::size_t N>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges,
             const tf::range<Iterator, N> &is_boundary,
             bool split_constraints = true) -> bool {
    clear();
    _converter = tf::make_exact_coordinate_converter<Int, Coord>(pts);

    // The caller's contract owns whether constraints may be split. The
    // no-split pass must still refuse on a crossing so the recovery wave
    // fires; only the wave's retry resolves incrementally and reports the
    // crossings it had to create.
    _allow_crossing_splits = split_constraints;

    if (!make_preserved_constraints(pts, edges, is_boundary))
      return false;

    return build_from_constraints();
  }

  /// Triangulate whatever constraint preparation left in _points and
  /// _final_constraint_edges.
  /// A constraint that cannot be recovered leaves the triangulation
  /// half-dismantled: the cavity around it is already unlinked and no labels
  /// have been computed. Reporting that state through the accessors would
  /// hand out triangles of both windings and a label for none of them, so a
  /// refused build reports nothing at all.
  auto abandon() -> void {
    _edges.clear();
    _region_labels.clear();
    _scratch_first_edge.clear();
    _scratch_tri_of_edge.clear();
    _scratch_parity.clear();
  }

  auto build_from_constraints() -> bool {
    auto n = static_cast<Index>(_points.size());
    if (n < 3)
      return true;

    _v_first_edge.allocate(n);
    topology::detail::fill_auto(_v_first_edge, k_none);
    _edges.reserve(static_cast<std::size_t>(6 * n));
    _flip_stack.reserve(static_cast<std::size_t>(6 * n));

    if (!build_delaunay(n)) {
      abandon();
      return false;
    }

    for (Index i = 0; i < static_cast<Index>(_final_constraint_edges.size());
         ++i) {
      auto edge = _final_constraint_edges[i];
      Index a = edge[0];
      Index b = edge[1];
      if (a == b)
        continue;
      if (!constrain_single_edge(a, b, _final_is_boundary[i] != 0,
                                 i < static_cast<Index>(
                                         _constraint_input_ids.size())
                                     ? _constraint_input_ids[std::size_t(i)]
                                     : k_none)) {
        abandon();
        return false;
      }
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
  // Order-free triangle extraction: a face is emitted from its minimum
  // half-edge. Chunked count + prefix + write, no per-face push_back.

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
      if (e1 == k_none)
        continue;
      Index e2 = prev_e(opp(e1));
      if (e2 == k_none || _edges[e1].boundary || _edges[e2].boundary)
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

    // One label per triangle, in the order make_faces() emits them. A
    // triangle the flood never reached is outside: reporting the unvisited
    // sentinel instead leaves the caller to read a parity out of it, and
    // -1 is even or odd depending on whether Index is signed.
    _region_labels.reserve(first_edge_of_tri.size());
    for (std::size_t t = 0; t < first_edge_of_tri.size(); ++t)
      _region_labels.push_back(parity[t] == k_none ? Index(0) : parity[t]);
  }

  // Lex-sort and exact-weld the converted points, then pass the
  // constraints through verbatim: a coordinate is one vertex, so a
  // constraint whose endpoints weld together has no length and is dropped.
  template <typename PointsPolicy, typename EdgesPolicy, typename Iterator,
            std::size_t N>
  auto make_preserved_constraints(const tf::points<PointsPolicy> &pts,
                                  const tf::edges<EdgesPolicy> &edges,
                                  const tf::range<Iterator, N> &is_boundary)
      -> bool {
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
    // The input id each constraint carries is what a discovered crossing
    // is reported against. Only the splitting pass resolves crossings, so
    // only it pays for the owner table.
    _constraint_input_ids.clear();
    if (_allow_crossing_splits)
      _constraint_input_ids.reserve(n_edges);
    // One flag per constraint. A shorter range would be read past its end,
    // so it is refused rather than half-consumed.
    if (static_cast<Index>(is_boundary.size()) < n_edges)
      return false;
    auto ib = is_boundary.begin();
    for (Index i = 0; i < n_edges; ++i, ++ib) {
      Index a = _index_map.f()[edges[i][0]];
      Index b = _index_map.f()[edges[i][1]];
      // Both endpoints welded onto one vertex: the constraint has no
      // length and nothing to recover, so it is dropped rather than
      // failed: a weld is not a reason to refuse a build.
      if (a == b)
        continue;
      _final_constraint_edges.push_back({a, b});
      _final_is_boundary.push_back(*ib ? 1 : 0);
      if (_allow_crossing_splits)
        _constraint_input_ids.push_back(i);
    }
    return true;
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

  // BRIO
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

    tf::topology::detail::build_insertion_order(_points, i, n,
                                               _insert_order, _scratch_keys);
    _locate_hint = find_any_interior_edge();

    for (Index k = 0; k < static_cast<Index>(_insert_order.size()); ++k) {
      insert_incremental(_insert_order[k]);
    }

    return true;
  }



  // Multiscale Hilbert sort (the BRIO recursion). Keeps the last 25% of the
  // range as its own Hilbert-sorted round, recursing on the first 75%.


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
  /// The triangle containing `query`, which is a vertex id or a bare
  /// coordinate — locate asks it for nothing but orientation.
  template <typename Query> auto locate(const Query &query) -> locate_result {
    Index e0 = _locate_hint;
    if (e0 == k_none || _edges[e0].boundary)
      e0 = find_any_interior_edge();

    int s0 = orient2d_sign(origin(e0), target(e0), query);
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

      int sa = orient2d_sign(origin(fa), target(fa), query);
      if (sa > 0) {
        if (_edges[opp(fa)].boundary)
          return {locate_kind::exterior, opp(fa)};
        e0 = opp(fa);
        s0 = -1;
        continue;
      }

      int sb = orient2d_sign(origin(fb), target(fb), query);
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
                                                           target(fe[k]), query)) {
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
    insert_incremental(p, locate(p));
  }

  /// The location is the caller's when it already had to find it: nothing
  /// changes the triangulation between deciding a point is new and placing
  /// it, so walking to the same triangle twice is the walk repeated.
  auto insert_incremental(Index p, locate_result loc) -> void {
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


  /// What stands between v0 and v1, if anything: a transversal crossing
  /// with a constrained edge, a vertex lying exactly on the constraint, or
  /// a constrained edge collinear with it. Only the first needs a new
  /// point; the other two are resolved on vertices that already exist,
  /// exactly.
  enum class obstruction_kind { none, crossing, vertex_on, collinear };
  struct obstruction {
    obstruction_kind kind = obstruction_kind::none;
    Index edge = k_none;
    Index vertex = k_none;
  };

  /// Walks from v0 toward v1 exactly as the cavity walk does, without
  /// mutating anything.
  auto find_obstruction(Index v0, Index v1) const -> obstruction {
    // A vertex lying on the constraint is reported by the walk that meets
    // it, in the order it is met, and by find_initial_triangle_for_constraint
    // when it is what stops the walk from starting. Scanning every point for
    // one instead costs a pass over the triangulation per constraint.
    Index initial = k_none, vertex_cw = k_none, vertex_ccw = k_none;
    Index blocking = k_none;
    if (!find_initial_triangle_for_constraint(v0, v1, initial, vertex_cw,
                                              vertex_ccw, &blocking)) {
      // A vertex sitting on the constraint stops the search before the
      // walk begins. It is the same obstruction the walk would report, so
      // report it — returning "nothing in the way" here is what turned a
      // resolvable case into a refusal.
      if (blocking != k_none)
        return {obstruction_kind::vertex_on, k_none, blocking};
      return {};
    }
    Index edge_across = prev_e(opp(initial));
    for (std::size_t guard = 0; guard < _edges.size() + 2; ++guard) {
      if (_edges[edge_across].constrained != k_unconstrained) {
        const Index c0 = origin(edge_across);
        const Index c1 = target(edge_across);
        if (orient2d_sign(v0, v1, c0) == 0 && orient2d_sign(v0, v1, c1) == 0)
          return {obstruction_kind::collinear, edge_across, k_none};
        return {obstruction_kind::crossing, edge_across, k_none};
      }
      Index edge_of_third = opp(prev_e(edge_across));
      Index third_vertex = origin(edge_of_third);
      if (third_vertex == v1)
        return {};
      int orientation = orient2d_sign(v0, v1, third_vertex);
      if (orientation == 0)
        return {obstruction_kind::vertex_on, k_none, third_vertex};
      edge_across = orientation < 0 ? opp(edge_of_third)
                                    : prev_e(edge_of_third);
    }
    return {};
  }

  /// Parameter of w along (a,b), exact: the projection preserves ratios,
  /// so ((w-a).u)/(u.u) is the parameter on the physical edge too.
  auto param_of_vertex(Index a, Index b, Index w) const -> param_t {
    using T2 = typename tf::exact::meta<Int>::T2;
    const auto pa = _points[std::size_t(a)];
    const auto pb = _points[std::size_t(b)];
    const auto pw = _points[std::size_t(w)];
    const T2 ux = T2(pb[0]) - T2(pa[0]), uy = T2(pb[1]) - T2(pa[1]);
    const T2 wx = T2(pw[0]) - T2(pa[0]), wy = T2(pw[1]) - T2(pa[1]);
    return crossing_param(wx * ux + wy * uy, ux * ux + uy * uy);
  }

  /// The crossing of (v0,v1) with (c0,c1), rounded to the lattice and
  /// appended as a new vertex. Returns k_none when the segments are
  /// parallel or the point does not land strictly inside both.
  /// Dyadic parameter of the crossing along (p0,p1), on the same 2^30 scale
  /// the split table uses. The parameter of a point along a segment is
  /// preserved by the affine 3D->2D projection, so a t computed here is
  /// valid for the physical edge in 3D.
  /// Strictly-interior parameter of a ratio along a segment; `-1` when the
  /// position is at or outside an endpoint, which is not a split.
  ///
  /// Interiority is decided on the EXACT ratio, never on the rounded
  /// parameter. A crossing closer to an endpoint than one parameter
  /// quantum still lies strictly inside the segment; judging it by the
  /// rounded value makes a real crossing invisible and the resolution
  /// refuse, which is a hole. The rounded parameter is a report, so it is
  /// only clamped into the interior.
  static auto crossing_param(typename tf::exact::meta<Int>::T2 num,
                             typename tf::exact::meta<Int>::T2 den)
      -> param_t {
    using T2 = typename tf::exact::meta<Int>::T2;
    if (den < T2(0)) {
      num = -num;
      den = -den;
    }
    if (num <= T2(0) || den <= num)
      return param_t(-1);
    const param_t t = tf::exact::dyadic_ratio<Int>(num, den);
    const param_t maximum = param_t(1) << k_crossing_param_bits;
    return t <= param_t(0) ? param_t(1)
                           : (t >= maximum ? maximum - param_t(1) : t);
  }

  /// Parameters of the crossing of (v0,v1) with (c0,c1), each on its own
  /// segment, as dyadic fractions of 2^k_crossing_param_bits. The
  /// parameter is the transportable fact; the coordinate is derived from
  /// it, never the other way round.
  auto crossing_params(Index v0, Index v1, Index c0, Index c1) -> bool {
    using T2 = typename tf::exact::meta<Int>::T2;
    const auto a = _points[std::size_t(v0)];
    const auto b = _points[std::size_t(v1)];
    const auto c = _points[std::size_t(c0)];
    const auto d = _points[std::size_t(c1)];
    const T2 ux = T2(b[0]) - T2(a[0]), uy = T2(b[1]) - T2(a[1]);
    const T2 vx = T2(d[0]) - T2(c[0]), vy = T2(d[1]) - T2(c[1]);
    const T2 denom = ux * vy - uy * vx;
    if (denom == T2(0))
      return false;
    const T2 wx = T2(c[0]) - T2(a[0]), wy = T2(c[1]) - T2(a[1]);
    const T2 num = wx * vy - wy * vx;
    _last_param_self = crossing_param(num, denom);
    _last_param_other = crossing_param(wx * uy - wy * ux, denom);
    return _last_param_self != param_t(-1) && _last_param_other != param_t(-1);
  }

  /// The dyadic blend the whole pipeline uses to place a split on an
  /// edge (`tf::cut::region_split_point`). Placing the crossing this way
  /// keeps it on the chord of the constraint being recovered and makes
  /// the coordinate a pure function of the parameter.
  auto blend_on(Index e0, Index e1, param_t parameter) const
      -> tf::point<Int, 2> {
    const auto a = _points[std::size_t(e0)];
    const auto b = _points[std::size_t(e1)];
    return {tf::exact::dyadic_blend(a[0], b[0], parameter),
            tf::exact::dyadic_blend(a[1], b[1], parameter)};
  }

  /// A crossing already resolved on the same physical constraint within
  /// one unit of 2^k_crossing_param_bits is the same crossing. Reusing it
  /// is what bounds repeated splitting — the exact integer form of
  /// `tf::cut::has_adjacent_region_split`, applied inside the build.
  /// A crossing already resolved AT THE SAME LATTICE POINT is the same
  /// crossing. Asked as an exact coordinate comparison rather than as a
  /// tolerance on the parameter: a parameter tolerance is measured in
  /// quanta, so on a long constraint every crossing near one end collapses
  /// to the same parameter and distinct crossings lose their identity.

  /// Returns k_none and names the coincident vertex when the crossing
  /// rounds onto one that already exists — the crossing is a T-junction
  /// there, not a failure.
  auto create_crossing_point(Index v0, Index v1, Index &coincident,
                             locate_result &located) -> Index {
    coincident = k_none;
    const auto q = blend_on(v0, v1, _last_param_self);
    // One coordinate is one vertex. A point that coincides with an existing
    // one is a corner of the triangle it would be inserted into, so the
    // triangle answers it in three comparisons — and answers it against every
    // vertex, not only against the crossings resolved before it.
    located = locate(q);
    if (located.he != k_none) {
      Index e = located.he;
      for (int corner = 0; corner < 3 && e != k_none; ++corner) {
        const auto candidate = _points[std::size_t(origin(e))];
        if (candidate[0] == q[0] && candidate[1] == q[1]) {
          coincident = origin(e);
          return k_none;
        }
        e = prev_e(opp(e));
      }
    }
    const Index p = static_cast<Index>(_points.size());
    _points.push_back(q);
    // kept_ids() must stay parallel to points(): consumers index it by
    // output point id, and `== f().size()` is the convention for "not an
    // input vertex". A resolution point has no input of its own.
    _index_map.kept_ids().push_back(static_cast<Index>(_index_map.f().size()));
    _v_first_edge.reallocate(_points.size());
    _v_first_edge[std::size_t(p)] = k_none;
    return p;
  }

  // The span is oriented per half-edge: `t0` belongs to `origin(he)` and
  // `t1` to `target(he)`, so the twin stores the reverse. A crossing is
  // measured along the half-edge the walk met, and rebasing it through a
  // span oriented the other way complements the parameter.
  auto note_constraint_owner(Index he, Index input_id, param_t t0,
                             param_t t1) -> void {
    if (input_id == k_none)
      return;
    if (_he_constraint.size() < _edges.size()) {
      const std::size_t old = _he_constraint.size();
      _he_constraint.reallocate(_edges.size());
      for (std::size_t i = old; i < _he_constraint.size(); ++i)
        _he_constraint[i] = {k_none, param_t(0), param_t(0)};
    }
    _he_constraint[std::size_t(he)] = {input_id, t0, t1};
    _he_constraint[std::size_t(opp(he))] = {input_id, t1, t0};
  }

  auto constraint_owner(Index he) const -> constraint_owner_t {
    return std::size_t(he) < _he_constraint.size()
               ? _he_constraint[std::size_t(he)]
               : constraint_owner_t{k_none, param_t(0), param_t(0)};
  }

  /// A parameter measured along a constraint PIECE, rebased onto the whole
  /// physical edge the piece belongs to. Without this the outer blend, which
  /// interpolates between the physical edge's endpoints, lands elsewhere.
  static auto rebase_param(param_t local, param_t t0, param_t t1)
      -> param_t {
    return tf::exact::rebase_parameter<Int>(
        local, t0, t1,
        typename tf::exact::meta<Int>::T2(1) << k_crossing_param_bits);
  }


  auto clear_constraint(Index he) -> void {
    _edges[he].constrained = k_unconstrained;
    _edges[opp(he)].constrained = k_unconstrained;
  }




  auto constrain_single_edge(
      Index v0, Index v1, bool is_boundary, Index input_id = k_none,
      param_t t0 = param_t(0),
      param_t t1 = param_t(1) << k_crossing_param_bits, int depth = 0,
      bool retried_from_far_end = false) -> bool {
    // Resolution recurses on the pieces it creates. A configuration that
    // keeps re-presenting the same obstruction must refuse cleanly rather
    // than spin — the recovery wave is what handles a refusal.
    if (depth > k_max_resolution_depth)
      return false;
    Index existing = edge_between(v0, v1);
    if (existing != k_none) {
      mark_constrained(existing, is_boundary);
      note_constraint_owner(existing, input_id, t0, t1);
      return true;
    }

    // A constraint that would cross an already inserted constraint is
    // resolved here rather than refused: the crossing becomes a vertex and
    // both constraints are replaced by their halves through it.
    if (const auto blocked = _allow_crossing_splits ? find_obstruction(v0, v1)
                                                   : obstruction{};
        blocked.kind != obstruction_kind::none) {
      // A vertex on the constraint, or the endpoint of a collinear
      // constrained edge lying inside it, splits it exactly — the vertex
      // is already there, so there is no coordinate to invent.
      if (blocked.kind != obstruction_kind::crossing) {
        Index w = blocked.vertex;
        if (w == k_none) {
          for (Index candidate : {origin(blocked.edge), target(blocked.edge)}) {
            if (candidate == v0 || candidate == v1)
              continue;
            const param_t t = param_of_vertex(v0, v1, candidate);
            if (t > param_t(0)) {
              w = candidate;
              break;
            }
          }
        }
        if (w == k_none) {
          return false;
        }
        const param_t t_w = rebase_param(param_of_vertex(v0, v1, w), t0, t1);
        if (t_w < param_t(0)) {
          return false;
        }
        if (input_id != k_none)
          _incremental_crossings.push_back(
              {w, input_id, k_none, t_w, param_t(0)});
        return constrain_single_edge(v0, w, is_boundary, input_id, t0, t_w,
                                     depth + 1) &&
               constrain_single_edge(w, v1, is_boundary, input_id, t_w, t1,
                                     depth + 1);
      }

      const Index crossed = blocked.edge;
      const Index c0 = origin(crossed);
      const Index c1 = target(crossed);
      const bool crossed_is_boundary =
          _edges[crossed].constrained == k_boundary_constrained;
      const auto other = constraint_owner(crossed);
      const Index other_id = other.input_id;
      if (!crossing_params(v0, v1, c0, c1))
        return false;
      // The parameter, rebased onto the physical edge, is what identifies
      // the crossing; the outer wave broadcasts it to every carrier.
      const param_t t_self = rebase_param(_last_param_self, t0, t1);
      const param_t t_other =
          rebase_param(_last_param_other, other.t0, other.t1);
      Index coincident = k_none;
      locate_result located{};
      Index p = create_crossing_point(v0, v1, coincident, located);
      if (p != k_none) {
        insert_incremental(p, located);
        if (input_id != k_none && other_id != k_none && input_id != other_id)
          _incremental_crossings.push_back(
              {p, input_id, other_id, t_self, t_other});
      }
      if (p == k_none) {
        if (coincident == k_none)
          return false;
        // The crossing rounds onto a vertex that already exists: split
        // whichever constraint does not already end there, exactly.
        if (coincident == c0 || coincident == c1) {
          const param_t t_w =
              rebase_param(param_of_vertex(v0, v1, coincident), t0, t1);
          if (t_w < param_t(0))
            return false;
          if (input_id != k_none)
            _incremental_crossings.push_back(
                {coincident, input_id, k_none, t_w, param_t(0)});
          return constrain_single_edge(v0, coincident, is_boundary, input_id,
                                       t0, t_w, depth + 1) &&
                 constrain_single_edge(coincident, v1, is_boundary, input_id,
                                       t_w, t1, depth + 1);
        }
        const param_t t_w = rebase_param(
            param_of_vertex(c0, c1, coincident), other.t0, other.t1);
        if (t_w < param_t(0))
          return false;
        if (Index stale = edge_between(c0, c1); stale != k_none)
          clear_constraint(stale);
        if (other_id != k_none)
          _incremental_crossings.push_back(
              {coincident, other_id, k_none, t_w, param_t(0)});
        return constrain_single_edge(c0, coincident, crossed_is_boundary,
                                     other_id, other.t0, t_w, depth + 1) &&
               constrain_single_edge(coincident, c1, crossed_is_boundary,
                                     other_id, t_w, other.t1, depth + 1) &&
               constrain_single_edge(v0, v1, is_boundary, input_id, t0, t1,
                                     depth + 1);
      }
      if (Index stale = edge_between(c0, c1); stale != k_none)
        clear_constraint(stale);
      return constrain_single_edge(c0, p, crossed_is_boundary, other_id,
                                   other.t0, t_other, depth + 1) &&
             constrain_single_edge(p, c1, crossed_is_boundary, other_id,
                                   t_other, other.t1, depth + 1) &&
             constrain_single_edge(v0, p, is_boundary, input_id, t0, t_self,
                                   depth + 1) &&
             constrain_single_edge(p, v1, is_boundary, input_id, t_self, t1,
                                   depth + 1);
    }

    Index initial = k_none;
    Index vertex_cw = k_none;
    Index vertex_ccw = k_none;
    Index blocking = k_none;
    if (!find_initial_triangle_for_constraint(v0, v1, initial, vertex_cw,
                                              vertex_ccw, &blocking)) {
      // A vertex lying on the constraint stops the search before the walk
      // starts. Resolving it is a split, so it is reported like every
      // other split and only ever made in resolve mode: a preserved build
      // must refuse instead, because refusing is what enters the region
      // into the recovery wave that broadcasts the split to every other
      // carrier of this edge. Splitting here silently conforms this
      // region alone and opens the seam against its neighbour.
      if (_allow_crossing_splits && blocking != k_none) {
        const param_t t_w =
            rebase_param(param_of_vertex(v0, v1, blocking), t0, t1);
        if (t_w >= param_t(0)) {
          if (input_id != k_none)
            _incremental_crossings.push_back(
                {blocking, input_id, k_none, t_w, param_t(0)});
          return constrain_single_edge(v0, blocking, is_boundary, input_id,
                                       t0, t_w, depth + 1) &&
                 constrain_single_edge(blocking, v1, is_boundary, input_id,
                                       t_w, t1, depth + 1);
        }
      }
      // The constraint is undirected: a fan that cannot start at v0 may
      // still start at v1. The reversed piece begins where the forward one
      // ends, so its span is the same two positions exchanged — mirroring
      // them instead reports every crossing at `maximum` minus its place.
      if (_allow_crossing_splits && !retried_from_far_end) {
        return constrain_single_edge(v1, v0, is_boundary, input_id, t1, t0,
                                     depth + 1, true);
      }
      return false;
    }

    _vertices_cw.clear();
    _vertices_ccw.clear();
    _deleted_edges.clear();

    if (!remove_inner_triangles(v0, v1, vertex_cw, vertex_ccw, initial)) {
      return false;
    }

    Index before_v0 = edge_between(v0, _vertices_ccw[1]);
    Index before_v1 =
        edge_between(v1, _vertices_cw[_vertices_cw.size() - std::size_t(2)]);

    Index constrained = create_edge_reusing(v0, v1, before_v0, before_v1,
                                            _deleted_edges.back());
    _deleted_edges.pop_back();
    mark_constrained(constrained, is_boundary);
    note_constraint_owner(constrained, input_id, t0, t1);

    if (!retriangulate_constraint_side(_vertices_cw, true))
      return false;
    if (!retriangulate_constraint_side(_vertices_ccw, false))
      return false;
    return true;
  }

  /// Is `p` strictly inside the segment `a`-`b`? `p` is a vertex or a bare
  /// coordinate, so the constraint walk and the point location can ask the
  /// one producer the same question.
  auto point_between_on_dominant_axis(Index a, Index b,
                                      const tf::point<Int, 2> &p) const
      -> bool {
    return tf::exact::is_between_on_segment<Int>(pt(a), pt(b), p);
  }

  auto point_between_on_dominant_axis(Index a, Index b, Index p) const
      -> bool {
    return tf::exact::is_between_on_segment<Int>(pt(a), pt(b), pt(p));
  }

  /// `blocking` names the vertex that lies ON the constraint when the
  /// search fails because of one. That is a resolvable obstruction, not a
  /// dead end, so it must reach the caller instead of being flattened
  /// into `false`.
  auto find_initial_triangle_for_constraint(Index v0, Index v1, Index &initial,
                                            Index &vertex_cw,
                                            Index &vertex_ccw,
                                            Index *blocking = nullptr) const
      -> bool {
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
          if (blocking != nullptr)
            *blocking = collinear;
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

    // Bounded like the read-only walk in find_obstruction: resolution
    // mutates the triangulation between recursive constraint insertions,
    // so a corrupt fan must refuse rather than spin. The recovery wave
    // is what handles a refusal.
    bool reached_v1 = false;
    for (std::size_t guard = 0; guard < _edges.size() + 2 && !reached_v1;
         ++guard) {
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

      if (reached) {
        reached_v1 = true;
        break;
      }

      edge_across = next_edge_across;
    }
    if (!reached_v1)
      return false;

    _vertices_cw.push_back(v1);
    _vertices_ccw.push_back(v1);
    if (_vertices_cw.size() < 3 || _vertices_ccw.size() < 3)
      return false;
    return true;
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

      // The ear loop pops as it consumes; below two vertices there is no
      // ear left to test and the indices would wrap.
      while (_retriangulation_stack.size() >= std::size_t(2)) {
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

  /// Against a coordinate that is not (yet) a vertex.
  auto orient2d_sign(Index ia, Index ib, const tf::point<Int, 2> &c) const
      -> int {
    return tf::exact::orient2d_sign(pt(ia), pt(ib), c);
  }


  // > 0 == d strictly inside the circumcircle of CCW a, b, c.
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
  tf::buffer<Index> _constraint_input_ids;
  param_t _last_param_self = param_t(-1);
  param_t _last_param_other = param_t(-1);
  bool _allow_crossing_splits = false;
  tf::buffer<constraint_owner_t> _he_constraint;
  tf::buffer<incremental_crossing> _incremental_crossings;
  tf::buffer<std::array<Index, 2>> _final_constraint_edges;
  tf::buffer<std::uint8_t> _final_is_boundary;

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


};

} // namespace tf
