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

namespace tf::exact {

/// Whether a floating-point filter's error bound holds in this build. A
/// fast-math build may reorder the filter's arithmetic, which voids the
/// bound, so every predicate skips its filter there and the exact path
/// decides alone.
inline constexpr bool float_filter_sound =
#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
    false;
#else
    true;
#endif

} // namespace tf::exact
