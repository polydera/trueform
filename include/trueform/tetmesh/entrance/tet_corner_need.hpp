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

namespace tf::tetmesh {

/// The frame depth the narrowest corner of a surface asks of its shells,
/// `corner_bits`, and that corner's angle in degrees, `min_angle`.
struct tet_corner_need {
  int corner_bits = 0;
  double min_angle = 180;
};

} // namespace tf::tetmesh
