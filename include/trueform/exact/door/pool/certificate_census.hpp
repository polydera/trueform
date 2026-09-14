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

#include <cstdint>

namespace tf::exact::door::pool {

/// What the certificate did and what it refused. There is no complete
/// enumeration behind the bounded solve, so `unavailable_frame`,
/// `solver_refusal` and `unadmitted` are refusals the SEARCH issued and not
/// proofs that no lattice point exists.
///
/// `narrow_solve` and `wide_solve` are the lane split: which rung answered,
/// counted where the answer was asked for and not where a frame was built,
/// because a frame is built once per direction and read by every vertex of
/// it. `shared_step` is the traffic neither of them saw — a vertex standing
/// at a residual its predecessor already solved.
struct certificate_census {
  std::int64_t calls = 0;
  std::int64_t zero_residual = 0;
  std::int64_t continuous_reject = 0;
  std::int64_t narrow_frames = 0;
  std::int64_t wide_frames = 0;
  std::int64_t narrow_solve = 0;
  std::int64_t wide_solve = 0;
  std::int64_t shared_step = 0;
  std::int64_t unavailable_frame = 0;
  std::int64_t solver_refusal = 0;
  std::int64_t unadmitted = 0;
  /// THE HOLE SPECIES: the plane stood within the band of the vertex — the
  /// continuous test passed — and the solve still found no lattice landing
  /// inside it. It is the lattice's own arithmetic and not a distance, and
  /// it is what the reachable-offset set predicts.
  std::int64_t lattice_holes = 0;
  std::int64_t certified = 0;

  auto bounded_refusals() const -> std::int64_t {
    return unavailable_frame + solver_refusal + unadmitted;
  }
  auto solves() const -> std::int64_t { return narrow_solve + wide_solve; }
};

/// A block's own tally added to the whole. Every counter is a count of
/// independent events, so the totals do not depend on the partition.
inline auto merge_certificate_census(const certificate_census &from,
                                     certificate_census &into) -> void {
  into.calls += from.calls;
  into.zero_residual += from.zero_residual;
  into.continuous_reject += from.continuous_reject;
  into.narrow_frames += from.narrow_frames;
  into.wide_frames += from.wide_frames;
  into.narrow_solve += from.narrow_solve;
  into.wide_solve += from.wide_solve;
  into.shared_step += from.shared_step;
  into.unavailable_frame += from.unavailable_frame;
  into.solver_refusal += from.solver_refusal;
  into.unadmitted += from.unadmitted;
  into.lattice_holes += from.lattice_holes;
  into.certified += from.certified;
}

} // namespace tf::exact::door::pool
