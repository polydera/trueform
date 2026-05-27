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

#include "../../exact/pt_converter.hpp"
#include "../../exact/vertex_converter.hpp"
#include "./predicate_kernel.hpp"

namespace tf::exact {

/// Build a kernel from a `pt_converter` and a world-coordinate tolerance.
/// Returns the default (exact) kernel for `tolerance <= 0`.
template <typename Int, typename Real, std::size_t Dims>
auto make_kernel(const pt_converter<Int, Real, Dims> &conv, double tolerance)
    -> predicate_kernel<Int> {
  if (tolerance <= 0.0)
    return {};
  return predicate_kernel<Int>(conv.convert_tolerance(Real(tolerance)));
}

/// Delegates to the `pt_converter` overload via `conv.coords`.
template <typename Int, typename Real, std::size_t Dims>
auto make_kernel(const vertex_converter<Int, Real, Dims> &conv,
                 double tolerance) -> predicate_kernel<Int> {
  return make_kernel(conv.coords, tolerance);
}

} // namespace tf::exact
