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

#include "./intersect_mode.hpp"

namespace tf {

/// Configuration for an intersection-detection run. `mode` selects the
/// classifier path; `tolerance` is the world-coordinate pitch the input's
/// planes are quantized to, and so the distance an input vertex may be
/// moved to stand on them — a statement about the input and not about a
/// predicate, stated in full by @ref tf::exact::input_lattice — and `0` is
/// the identity. Implicitly constructible from `intersect_mode`.
struct intersect_config {
  intersect_mode mode = intersect_mode::primitives;
  double tolerance = 0.0;

  constexpr intersect_config() = default;
  constexpr intersect_config(intersect_mode m) : mode(m) {}
  constexpr intersect_config(intersect_mode m, double tol)
      : mode(m), tolerance(tol) {}
};

} // namespace tf
