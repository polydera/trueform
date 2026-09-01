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
#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/index_map.hpp"
#include "../../core/none.hpp"
#include "../../core/points.hpp"
#include "../../core/range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../exact/resolve_int_type.hpp"
#include "../cdt_refine_config.hpp"
#include "../cdt_region_mode.hpp"
#include "./build_constrained_delaunay_refinement.hpp"
#include "./clear_constrained_delaunay_refinement.hpp"
#include "./constrained_delaunay_refinement_face_constraint_owners.hpp"
#include "./constrained_delaunay_refinement_owner.hpp"
#include "./for_each_constrained_delaunay_refinement_face.hpp"
#include "./for_each_constrained_delaunay_refinement_steiner.hpp"
#include <array>
#include <cstddef>

namespace tf {

template <typename Index, typename Coord, typename Int> class cdt_refiner;

} // namespace tf

namespace tf::topology::cdt {

template <typename Index, typename Coord,
          typename Int = tf::exact::resolve_int_type<tf::none_t, Coord>>
class constrained_delaunay_refinement_producer {
  using owner_type =
      constrained_delaunay_refinement_owner<Index, Coord, Int>;

  auto owner() -> owner_type & { return _owner; }
  auto owner() const -> const owner_type & { return _owner; }

  owner_type _owner;

  friend class ::tf::cdt_refiner<Index, Coord, Int>;

public:
  using param_t = typename owner_type::param_type;
  using constraint_owner_t = typename owner_type::constraint_provenance;
  using constraint_collision = typename owner_type::constraint_collision;

  static constexpr int k_crossing_param_bits = owner_type::crossing_param_bits;
  static constexpr int crossing_param_bits() { return k_crossing_param_bits; }

  auto clear() -> void { clear_constrained_delaunay_refinement(owner()); }

  template <typename PointsPolicy, typename EdgesPolicy,
            typename BoundaryIterator, std::size_t BoundaryN,
            typename SplittableIterator, std::size_t SplittableN>
  auto build(const tf::points<PointsPolicy> &points,
             const tf::edges<EdgesPolicy> &edges,
             const tf::range<BoundaryIterator, BoundaryN> &is_boundary,
             const tf::range<SplittableIterator, SplittableN> &is_splittable,
             const tf::cdt_refine_config &config,
             tf::cdt_region_mode mode = tf::cdt_region_mode::nesting) -> bool {
    return build_constrained_delaunay_refinement(
        owner(), points, edges, is_boundary, is_splittable, config, mode);
  }

  /// The fact the adopted labels carry, owned by the seed CDT.
  auto region_mode() const -> tf::cdt_region_mode {
    return owner()._cdt.region_mode();
  }

  auto ok() const -> bool { return owner()._ok; }
  auto n_input_points() const -> Index { return owner()._n_input_points; }
  auto n_input_constraints() const -> Index { return owner()._n_input_edges; }

  auto has_prepared_input_index_map() const -> bool {
    return owner()._cdt.has_prepared_input_index_map();
  }

  auto index_map() const -> const tf::index_map_buffer<Index> & {
    return owner()._cdt.index_map();
  }

  auto constraint_splits() const {
    return tf::make_offset_block_range(owner()._split_offsets,
                                       owner()._splits);
  }

  auto n_constraint_splits() const -> Index {
    return static_cast<Index>(owner()._splits.size());
  }

  /// Visit the refined faces with full adjacency: f(i, v0, v1, v2, label,
  /// nbrs, cons), nbrs[k] the face across edge (vk, vk+1) or the sentinel,
  /// cons[k] whether that edge is a constraint. Face ids and order match
  /// `region_labels()` and the materialized faces.
  template <typename F> auto for_each_face_adjacency(F &&f) const -> void {
    for_each_constrained_delaunay_refinement_face(owner(),
                                                  static_cast<F &&>(f));
  }

  /// The input constraint each edge of face `triangle` belongs to and the
  /// span it covers, oriented from corner k to corner k + 1.
  auto face_constraint_owners(Index triangle) const
      -> std::array<constraint_owner_t, 3> {
    return constrained_delaunay_refinement_face_constraint_owners(owner(),
                                                                  triangle);
  }

  /// Retain constraint ownership on the seed CDT, which the collision report
  /// is stated against. The setting persists across `clear()` and builds.
  auto always_track_constraint_owners() -> void {
    owner()._cdt.always_track_constraint_owners();
  }

  /// Seed recoveries that found their edge already claimed by a DIFFERENT
  /// input constraint. Recorded only when the seed build tracks constraint
  /// owners.
  auto constraint_collisions() const
      -> const tf::buffer<constraint_collision> & {
    return owner()._cdt.constraint_collisions();
  }

  template <typename F>
  auto for_each_interior_steiner_point(F &&f) -> void {
    for_each_constrained_delaunay_refinement_steiner(
        owner(), static_cast<F &&>(f));
  }
};

} // namespace tf::topology::cdt
