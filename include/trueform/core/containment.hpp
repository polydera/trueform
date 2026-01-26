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
/// @brief Containment relationship between point and polygon.
///
/// Returned by @ref tf::classify when testing whether a point lies
/// inside, outside, or on the boundary of a polygon.
enum class containment { inside = 0, outside = 1, on_boundary = 2 };

/// @ingroup core
/// @brief Strict containment without boundary case.
///
/// Used when boundary cases are handled separately or not relevant.
enum class strict_containment { inside = 0, outside = 1 };

} // namespace tf
