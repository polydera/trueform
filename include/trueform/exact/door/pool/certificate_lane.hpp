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

#include "./exact_lane.hpp"
#include "./exact_plane_frame.hpp"

#include "../plane_frame.hpp"

#include <array>

namespace tf::exact::door::pool {

/// The step onto the plane a residual already bought.
///
/// A shift is a function of the frame and the residual alone, so two
/// vertices standing the same distance off the plane take the same step onto
/// it — and a planar wall states ONE residual for its whole support.
///
/// What stays the vertex's own is the sum: the lattice range and the band
/// are facts about `original + shift` and never about the shift.
template <typename Int> struct solved_step {
  typename exact_lane<Int>::coefficient_type residual{};
  std::array<typename exact_lane<Int>::coefficient_type, 3> shift{};
  bool solved = false;
  bool held = false;
};

/// One exact direction's two frames and which of them the rung admits.
///
/// A frame is a function of the primitive normal alone, so one direction
/// costs one Lagrange reduction however many planes and vertices read it,
/// and the lane is held BY THE CALLER against its own dense direction
/// space — a certificate never looks a direction up.
///
/// NEITHER FRAME IS BUILT IN ADVANCE. A residual of zero and a residual
/// beyond the continuous reach are answered without any frame at all, so a
/// direction that never solves never pays a reduction on either rung.
template <typename Int> struct certificate_lane {
  tf::exact::door::plane_frame<Int> narrow{};
  exact_plane_frame<Int> wide{};
  bool narrow_available = false;
  bool narrow_stated = false;
  bool wide_stated = false;
};

} // namespace tf::exact::door::pool
