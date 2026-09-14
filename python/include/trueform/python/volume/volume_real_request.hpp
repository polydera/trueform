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

namespace tf::py {

/// The coordinate type a volume call is asked to emit in, crossing as a code
/// the facade validated — the same shape the isosurface method and the boolean
/// operation cross in. This is the one place the code names a type, so every
/// entry answers the same dtype with the same instantiation.
///
/// @param f Called as `f(Real{})`, and must return the same Python object type
///        for every supported real.
template <typename F> auto with_requested_real(int dtype, F &&f) {
  if (dtype == 1)
    return f(double{});
  return f(float{});
}

} // namespace tf::py
