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
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/edges.hpp"
#include "../core/index_map.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/points.hpp"
#include "../core/range.hpp"
#include "../core/views/constant.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../exact/resolve_int_type.hpp"
#include "./cdt/constrained_delaunay_refinement_owner.hpp"
#include "./cdt/constrained_delaunay_refinement_producer.hpp"
#include "./cdt/materialize_constrained_delaunay_refinement_faces.hpp"
#include "./cdt_constraint_split.hpp"
#include "./cdt_refine_config.hpp"
#include "./cdt_region_mode.hpp"
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup topology
/// @brief Incremental Delaunay refinement over the constraint-preserving CDT.
///
/// The boundary CDT is built once (constraints never intersected) and adopted
/// into a persistent triangle structure; a bad-triangle queue then drives
/// circumcenter insertions directly -- no rebuild rounds and no point
/// location, since every candidate walks from its generating triangle.
/// Triangles are judged by the quality measure q = (2/sqrt3) * 2A /
/// max_edge^2. Constrained edges are never flipped; a candidate that
/// encroaches a constraint's diametral circle splits the constraint at its
/// midpoint instead (Ruppert), as does an existing vertex inside a diametral
/// circle. Vertices constraint-connected to a segment's endpoint are exempt
/// encroachers (sharp input corners would otherwise split forever), and each
/// input constraint splits at most k_max_split_depth generations.
///
/// Every split is recorded as a dyadic parameter of the ORIGINAL input edge
/// (see @ref cdt_constraint_split), so callers coordinating several
/// triangulations over shared polylines can union the records and reproduce
/// identical split points. Steiner positions are deterministic for identical
/// input.
///
/// Constraints whose endpoints become one prepared point are omitted. Build
/// returns false (bail, topology and refinement products unusable) when a
/// remaining constraint cannot be recovered under the preserve contract.
/// Exact input preparation may already have published its index map; query
/// `has_prepared_input_index_map()` before consuming that retained subproduct.
///
/// @tparam Index The index type.
/// @tparam Coord The input coordinate type; integral coordinates are consumed
///   exactly (identity conversion). To make dyadic split positions exact on
///   an integer lattice, reserve k_max_split_depth bits of scale headroom.
/// @tparam Int The integer type for exact predicates; defaults per Coord
///   via @ref tf::exact::resolve_int_type (int32 for float, int64 for
///   double, identity for integral Coord).
template <typename Index, typename Coord,
          typename Int = tf::exact::resolve_int_type<tf::none_t, Coord>>
class cdt_refiner {
  using producer_type =
      tf::topology::cdt::constrained_delaunay_refinement_producer<Index, Coord,
                                                                  Int>;
  using owner_type =
      tf::topology::cdt::constrained_delaunay_refinement_owner<Index, Coord,
                                                               Int>;

  auto owner() -> owner_type & { return _producer.owner(); }
  auto owner() const -> const owner_type & { return _producer.owner(); }

  producer_type _producer;

public:
  /// A constraint span is stated on this scale: the whole input constraint is
  /// [0, 1 << crossing_param_bits()], the same scale
  /// @ref tf::constrained_delaunay_triangulator states its own owners on.
  using param_t = typename producer_type::param_t;
  using constraint_owner_t = typename producer_type::constraint_owner_t;
  using constraint_collision = typename producer_type::constraint_collision;

  static constexpr int k_crossing_param_bits =
      producer_type::k_crossing_param_bits;
  static constexpr int crossing_param_bits() { return k_crossing_param_bits; }

  auto clear() -> void { _producer.clear(); }

  template <typename PointsPolicy, typename EdgesPolicy, typename Iterator,
            std::size_t N>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges,
             const tf::range<Iterator, N> &is_boundary,
             const tf::cdt_refine_config &config = {},
             tf::cdt_region_mode mode = tf::cdt_region_mode::nesting) -> bool {
    return build(pts, edges, is_boundary,
                 tf::make_constant_range(true, edges.size()), config, mode);
  }

  /// @brief Build with a per-edge `is_splittable` mask. Splittability belongs
  /// to the physical constraint after input-point welding: coincident inputs
  /// union the flag, and every input alias inherits the physical edge's split
  /// sequence in its own orientation. A physical constraint whose aliases are
  /// all non-splittable is born at the split-depth cap.
  template <typename PointsPolicy, typename EdgesPolicy,
            typename BoundaryIterator, std::size_t BoundaryN,
            typename SplittableIterator, std::size_t SplittableN>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges,
             const tf::range<BoundaryIterator, BoundaryN> &is_boundary,
             const tf::range<SplittableIterator, SplittableN> &is_splittable,
             const tf::cdt_refine_config &config,
             tf::cdt_region_mode mode = tf::cdt_region_mode::nesting) -> bool {
    return _producer.build(pts, edges, is_boundary, is_splittable, config,
                           mode);
  }

  template <typename PointsPolicy, typename EdgesPolicy>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges,
             const tf::cdt_refine_config &config = {},
             tf::cdt_region_mode mode = tf::cdt_region_mode::nesting) -> bool {
    return build(pts, edges, tf::make_constant_range(true, edges.size()),
                 config, mode);
  }

  auto ok() const -> bool { return _producer.ok(); }
  auto n_input_points() const -> Index { return _producer.n_input_points(); }

  /// @brief Whether exact input preparation published the authoritative input
  /// index map. Constraint recovery may refuse after this point; the topology
  /// result is then unusable, but this prepared identity product remains valid.
  auto has_prepared_input_index_map() const -> bool {
    return _producer.has_prepared_input_index_map();
  }

  /// @brief Number of output faces; face ids align with region_labels.
  auto n_faces() const -> Index {
    return static_cast<Index>(owner()._t.size());
  }

  /// @brief Face corners by face id (no allocation; region_labels order).
  auto face(Index i) const -> std::array<Index, 3> {
    const auto &triangle = owner()._t[std::size_t(i)];
    return {triangle.v[0], triangle.v[1], triangle.v[2]};
  }

  auto make_faces() const -> tf::blocked_buffer<Index, 3> {
    return topology::cdt::materialize_constrained_delaunay_refinement_faces(
        owner());
  }

  /// @brief Per-face region labels (match make_faces order). The build's
  /// `tf::cdt_region_mode` says which fact they are; the refinement adopts
  /// them from its seed triangulation and carries them through every split.
  auto region_labels() const { return tf::make_range(owner()._label); }

  /// The fact currently carried by `region_labels()`.
  auto region_mode() const -> cdt_region_mode {
    return _producer.region_mode();
  }

  /// @brief Visit faces with full adjacency: f(i, v0, v1, v2, label, nbrs,
  /// cons), nbrs[k] the face across edge (vk, vk+1) or `Index(-1)`, cons[k]
  /// whether that edge is a constraint. Face ids and order match
  /// `make_faces()` and `region_labels()`.
  template <typename F> auto for_each_face_adjacency(F &&f) const -> void {
    _producer.for_each_face_adjacency(static_cast<F &&>(f));
  }

  /// @brief The full owner record for each edge of face `triangle` — the
  /// input constraint ID and the exact span this edge covers of it, oriented
  /// from corner k to corner k + 1. A split sub-edge states its ORIGINAL
  /// input constraint, on that constraint's whole scale.
  auto face_constraint_owners(Index triangle) const
      -> std::array<constraint_owner_t, 3> {
    return _producer.face_constraint_owners(triangle);
  }

  /// Retain constraint ownership on the seed triangulation, which the
  /// collision report is stated against. The setting persists across
  /// `clear()` and builds.
  auto always_track_constraint_owners() -> void {
    _producer.always_track_constraint_owners();
  }

  /// @brief Seed recoveries that found their edge already claimed by a
  /// DIFFERENT input constraint, as {edge, prior_input, input}: the two
  /// inputs cover one span. Recorded only when the seed build tracks
  /// constraint owners.
  auto constraint_collisions() const
      -> const tf::buffer<constraint_collision> & {
    return _producer.constraint_collisions();
  }

  /// @brief Output points on the internal integer lattice.
  auto points() const { return tf::make_points(owner()._ip); }

  /// @brief Output points deconverted to Coord.
  auto converted_points() const {
    if constexpr (std::is_integral_v<Coord>) {
      return tf::make_points(
          tf::make_mapped_range(owner()._ip, [](const auto &point) {
            return tf::point<Coord, 2>{Coord(point[0]), Coord(point[1])};
          }));
    } else {
      return tf::make_points(
          tf::make_mapped_range(owner()._dp, [](const auto &point) {
            return tf::point<Coord, 2>{Coord(point[0]), Coord(point[1])};
          }));
    }
  }

  /// @brief Index map between input point space and prepared output point
  /// space. After a refused build it is valid only when
  /// `has_prepared_input_index_map()` returns true.
  auto index_map() const -> const tf::index_map_buffer<Index> & {
    return _producer.index_map();
  }

  /// @brief Per input constraint edge, its recorded splits sorted by
  /// parameter. Indexed by the input edge id.
  auto constraint_splits() const {
    return _producer.constraint_splits();
  }

  auto n_constraint_splits() const -> Index {
    return _producer.n_constraint_splits();
  }

  /// @brief Is edge `e` of output face `f` (make_faces order) constrained?
  auto constrained(Index f, int e) const -> bool {
    return owner().constrained(f, e);
  }
};

} // namespace tf
