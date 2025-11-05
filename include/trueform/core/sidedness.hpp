/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {
enum class sidedness {
  on_positive_side =
      0, // Corresponds to 'above' a plane or 'right_of' a 2D segment
  on_negative_side =
      1,          // Corresponds to 'below' a plane or 'left_of' a 2D segment
  on_boundary = 2 // The point is coplanar or colinear
};
}
