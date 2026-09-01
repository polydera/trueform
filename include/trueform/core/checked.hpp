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

namespace tf {

/// @ingroup core
/// @brief Tag for the primitives' serial fallback, carrying its cutoff.
///
/// `tf::checked` selects the default; `tf::checked(n)` states the
/// range length below which the serial path runs — for call sites
/// whose per-element work sits far from the convention's assumption.
struct checked_t {
  unsigned long serial_below = 1000;
  constexpr auto operator()(unsigned long n) const -> checked_t {
    return {n};
  }
};

/// @ingroup core
/// @brief Tag instance for checked execution.
static constexpr checked_t checked;

} // namespace tf
