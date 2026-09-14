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

#include "./certificate_census.hpp"
#include "./certificate_lane.hpp"
#include "./exact_dot.hpp"
#include "./exact_lane.hpp"
#include "./exact_plane_frame.hpp"
#include "./make_narrow_plane_frame.hpp"
#include "./narrow_lane_bound.hpp"
#include "./place_on_exact_plane.hpp"
#include "./tally_census.hpp"

#include "../../../core/point.hpp"
#include "../../canonical_plane.hpp"
#include "../../meta.hpp"
#include "../admits_placement.hpp"
#include "../place_on_plane.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace tf::exact::door::pool {

/// Where a vertex stands on an exact plane, or the refusal that it does
/// not stand on it within the band.
template <typename Int> struct exact_certificate {
  tf::point<Int, 3> point{};
  bool certified = false;
};

/// Whether `original` has a lattice point of the exact plane `plane` within
/// `tolerance`, and where.
///
/// A zero residual is answered at the point itself, before any frame is
/// asked for: an original vertex of the plane's own support moves NOWHERE
/// and pays one dot product for the fact. A residual beyond the continuous
/// reach is refused by the same one dot product squared. Only what survives
/// both reaches the bounded rank-1 solve.
///
/// THE SOLVE HAS TWO LANES. The NARROW lane runs the door's own quantized
/// rung when this direction's whole frame workspace and this vertex's own
/// step onto the plane stand inside it — which a gcd-reduced simple normal
/// does at any face size. The WIDE lane is the product rung and carries
/// every direction the narrow one refuses. Both run the same algorithm from
/// the same lift, each reduced on its own rung, and both end ON the plane
/// and inside the band; the lane moves the price and which admissible
/// lattice point comes back, and it is chosen by WIDTH and never by where
/// the plane came from.
///
/// The solve is the certificate's whole authority: a refusal is recorded as
/// the SEARCH's refusal and the caller takes it as "no pool", which is
/// always safe. Nothing here proves lattice infeasibility.
///
/// A CERTIFIED POINT IS ON ITS PLANE BY AN INTEGER IDENTITY, so nothing
/// asks it again. A name is primitive, so its Bezout vector carries
/// `N . s == 1` exactly; the kernel basis the reduction and the sweep walk
/// carries `N . b == 0`; and every step of both refuses rather than wraps.
/// The shift therefore lands the residual on zero, and the only questions
/// left are the vertex's own — the lattice range and the band.
///
/// `step` carries the solve the caller's previous vertex already bought
/// (@ref tf::exact::door::pool::solved_step). It is a memo of the arithmetic
/// and never of the verdict: the band and the lattice range are asked of
/// every vertex on its own sum.
template <typename Int>
auto certify_exact(const tf::exact::canonical_plane<Int> &plane,
                   const tf::point<Int, 3> &original, Int tolerance,
                   certificate_lane<Int> &lane, solved_step<Int> &step,
                   certificate_census &census) -> exact_certificate<Int> {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  using product_type = typename exact_lane<Int>::product_type;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  tally_census(census.calls);
  exact_certificate<Int> certificate{original, false};

  const coefficient_type r =
      -tf::exact::orient3d_plane_value<Int>(plane, original);
  if (r == coefficient_type(0)) {
    tally_census(census.zero_residual);
    certificate.certified = true;
    tally_census(census.certified);
    return certificate;
  }

  const std::array<coefficient_type, 3> normal{plane[0], plane[1], plane[2]};
  const product_type band(tolerance);
  if (product_type(r) * product_type(r) >
      band * band * exact_dot<Int>(normal, normal)) {
    tally_census(census.continuous_reject);
    return certificate;
  }

  if (step.held && step.residual == r) {
    tally_census(census.shared_step);
    if (!step.solved)
      return certificate;
  } else {
    step.held = true;
    step.residual = r;
    step.solved = false;
    if (!lane.narrow_stated) {
      tally_census(census.narrow_frames);
      lane.narrow_available = make_narrow_plane_frame<Int>(normal, lane.narrow);
      lane.narrow_stated = true;
    }
    const T1 reach = narrow_lane_bound<Int>();
    bool narrow = lane.narrow_available && r <= coefficient_type(reach) &&
                  r >= -coefficient_type(reach);
    const T1 residual = narrow ? static_cast<T1>(r) : T1(0);
    for (std::size_t k = 0; k < 3 && narrow; ++k) {
      const T2 onto = T2(residual) * T2(lane.narrow.s[k]);
      narrow = onto <= T2(reach) && onto >= -T2(reach);
    }

    if (narrow) {
      tally_census(census.narrow_solve);
      std::array<T1, 3> shift{};
      if (!tf::exact::door::place_on_plane<Int>(lane.narrow, residual, shift)) {
        tally_census(census.solver_refusal);
        return certificate;
      }
      for (std::size_t k = 0; k < 3; ++k)
        step.shift[k] = coefficient_type(shift[k]);
    } else {
      tally_census(census.wide_solve);
      if (!lane.wide_stated) {
        tally_census(census.wide_frames);
        lane.wide = make_exact_plane_frame<Int>(normal);
        lane.wide_stated = true;
      }
      if (!lane.wide.available) {
        tally_census(census.unavailable_frame);
        return certificate;
      }
      std::array<coefficient_type, 3> shift{};
      if (!place_on_exact_plane<Int>(lane.wide, r, shift)) {
        tally_census(census.solver_refusal);
        return certificate;
      }
      step.shift = shift;
    }
    step.solved = true;
  }

  std::array<T1, 3> placed{};
  const coefficient_type ceiling(std::numeric_limits<Int>::max());
  const coefficient_type floor(std::numeric_limits<Int>::min());
  for (std::size_t k = 0; k < 3; ++k) {
    const coefficient_type at = coefficient_type(original[k]) + step.shift[k];
    if (at > ceiling || at < floor) {
      tally_census(census.unadmitted);
      return certificate;
    }
    placed[k] = static_cast<T1>(at);
  }

  if (!tf::exact::door::admits_placement<Int>(original, placed, tolerance)) {
    tally_census(census.unadmitted);
    // the continuous reach already admitted this plane, so the only thing
    // that refused is where the lattice put the landing
    tally_census(census.lattice_holes);
    return certificate;
  }
  certificate.point = tf::point<Int, 3>{static_cast<Int>(placed[0]),
                                        static_cast<Int>(placed[1]),
                                        static_cast<Int>(placed[2])};
  certificate.certified = true;
  tally_census(census.certified);
  return certificate;
}

} // namespace tf::exact::door::pool
