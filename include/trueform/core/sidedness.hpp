/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {
enum class sidedness {
  on_positive_side, // Corresponds to 'above' a plane or 'right_of' a 2D segment
  on_negative_side, // Corresponds to 'below' a plane or 'left_of' a 2D segment
  on_boundary       // The point is coplanar or colinear
};
}
