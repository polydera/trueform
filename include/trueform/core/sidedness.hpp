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
/// @brief Classification of point position relative to plane or line.
///
/// Returned by @ref tf::classify when testing point position relative
/// to oriented primitives (planes, lines, rays, segments).
///
/// - `on_positive_side`: Above a plane or right of a 2D line/segment
/// - `on_negative_side`: Below a plane or left of a 2D line/segment
/// - `on_boundary`: Coplanar or collinear with the primitive
enum class sidedness {
  on_positive_side = 0,
  on_negative_side = 1,
  on_boundary = 2
};

} // namespace tf
