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

#include "../../../core/buffer.hpp"
#include "../../../core/point.hpp"
#include "../../canonical_plane.hpp"

#include <array>
#include <cstdint>

namespace tf::exact::door::pool {

/// A face speaking for the exact plane it stands on: the names that must
/// all resolve to that plane for the face to be realized on it — its own
/// name and the names its three corners consume as their target source —
/// and the corners themselves. `-1` pads the tuple.
///
/// The support triple is sorted, so a face names one witness whatever its
/// winding, and that sorted triple is the witness's own election key.
///
/// `plane_name` is the id of the face's own plane in the registry, not a
/// plane.
template <typename Int> struct pool_witness {
  std::array<tf::point<Int, 3>, 3> support{};
  std::array<int, 4> name{-1, -1, -1, -1};
  int plane_name = -1;
};

/// The immutable input the partition is a pure function of.
///
/// A NAME IS AN EXACT SUPPORT PLANE. `plane` is the canonical registry —
/// primitive, sign-fixed, deduplicated and sorted — so a name's id IS its
/// position in canonical order, equal normals occupy a contiguous id run,
/// and inside that run the ids ascend by offset. Every later "canonical
/// order" is therefore an integer comparison.
///
/// `support` is `R(n)`: every original nonfeature vertex the name carries,
/// deduplicated, one contiguous block per name. A name is indivisible — one
/// distant occurrence refuses it globally.
///
/// `support_point` is a DIFFERENT fact and the face-plane tier owns it: the
/// original vertex of the name's own supported faces least by (squared
/// lattice norm, lexicographic coordinates), FEATURE FACES INCLUDED. `R(n)`
/// can be empty on a valid name — every face of it may carry a feature
/// corner — so the ray a pool is described along may not be read off it.
template <typename Int> struct pool_names {
  tf::buffer<tf::exact::canonical_plane<Int>> plane;
  tf::buffer<int> support_offsets;
  tf::buffer<tf::point<Int, 3>> support;
  tf::buffer<tf::point<Int, 3>> support_point;
  tf::buffer<pool_witness<Int>> witness;
};

/// Where each name ended up. `-1` in `pool_of_name` is the name no exact
/// election committed, which is not the same fact as being alone in its
/// family.
template <typename Int> struct pool_partition {
  tf::buffer<int> pool_of_name;
  tf::buffer<tf::exact::canonical_plane<Int>> pool_plane;
};

/// What the partition did and what it spent.
///
/// THE CONSERVATIVE MISS IS THE CHART'S ALONE. `cell_straddle_exposure`
/// counts names standing in a cell that has an occupied neighbour, which is
/// where a fixed boundary can separate a feasible pair. The ladder has no
/// miss of its own: its walk reaches every entry the proposal band covers,
/// so everything else a name fails to join, it failed to certify. The count
/// is the exclusion and not a proof that a feasible weld was lost.
///
/// NOTHING HERE IS ON THE DOOR'S PATH. Every tally goes through
/// @ref tf::exact::door::pool::tally_census and compiles away without
/// `TF_POOL_CENSUS`; every fact a phase would have to WORK to state — the
/// straddle scan, the run extents, the pool histogram, the phase clocks —
/// is guarded at the work. What remains unguarded is a phase storing an
/// extent it is already holding.
///
/// `gap_filtered` is what the double screen decided, `gap_exact` what the
/// width proof carried on the product rung, and `gap_limbs` what needed the
/// limb scratch.
struct election_census {
  std::int64_t names = 0;
  std::int64_t cells = 0;
  std::int64_t speaking_cells = 0;
  std::int64_t witnesses = 0;

  std::int64_t cell_straddle_exposure = 0;

  std::int64_t runs = 0;
  std::int64_t longest_run = 0;
  std::int64_t gap_cuts = 0;
  std::int64_t anchors = 0;
  std::int64_t anchors_without_witness = 0;
  std::int64_t anchors_realizing_nothing = 0;
  std::int64_t extended_members = 0;
  std::int64_t consumed_refusals = 0;
  std::int64_t cores_without_admissible = 0;
  std::int64_t partnerless_anchors = 0;
  std::int64_t runs_without_anchor = 0;
  std::int64_t certificate_queries = 0;
  std::int64_t empty_supports = 0;
  std::int64_t pools = 0;
  std::int64_t exact_singletons = 0;
  std::int64_t pooled_names = 0;
  std::int64_t unpooled_names = 0;
  std::int64_t redirected_vertices = 0;

  std::int64_t gap_tests = 0;
  std::int64_t gap_filtered = 0;
  std::int64_t gap_exact = 0;
  std::int64_t gap_limbs = 0;

  double straddle_seconds = 0.0;
  double scene_seconds = 0.0;
  double witness_seconds = 0.0;
  double cell_seconds = 0.0;
  double ladder_seconds = 0.0;
  double walk_seconds = 0.0;
  double publish_seconds = 0.0;

  auto trials() const -> std::int64_t { return certificate_queries; }
  /// The partition's own phases. `scene_seconds` is the naming pass before
  /// them and is not one of its terms.
  auto seconds() const -> double {
    return witness_seconds + cell_seconds + ladder_seconds + walk_seconds +
           publish_seconds;
  }
};

/// A block's own tally added to the whole. Every counter merged here counts
/// independent events, so the totals do not depend on the partition; the
/// extents a phase publishes (`names`, `cells`, `witnesses`, `runs`,
/// `longest_run`, `speaking_cells`, `gap_cuts`) are their producer's own
/// facts and are not summed.
inline auto merge_election_census(const election_census &from,
                                  election_census &into) -> void {
  into.cell_straddle_exposure += from.cell_straddle_exposure;
  into.straddle_seconds += from.straddle_seconds;
  into.anchors += from.anchors;
  into.anchors_without_witness += from.anchors_without_witness;
  into.anchors_realizing_nothing += from.anchors_realizing_nothing;
  into.extended_members += from.extended_members;
  into.consumed_refusals += from.consumed_refusals;
  into.cores_without_admissible += from.cores_without_admissible;
  into.partnerless_anchors += from.partnerless_anchors;
  into.runs_without_anchor += from.runs_without_anchor;
  into.certificate_queries += from.certificate_queries;
  into.empty_supports += from.empty_supports;
  into.exact_singletons += from.exact_singletons;
  into.pooled_names += from.pooled_names;
  into.unpooled_names += from.unpooled_names;
  into.redirected_vertices += from.redirected_vertices;
  into.gap_tests += from.gap_tests;
  into.gap_filtered += from.gap_filtered;
  into.gap_exact += from.gap_exact;
  into.gap_limbs += from.gap_limbs;
}

} // namespace tf::exact::door::pool
