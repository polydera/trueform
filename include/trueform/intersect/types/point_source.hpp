/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf::intersect {
enum class point_source : char {
  other = 0,
  original = 1,
  own = 1 << 2,
  created = 1 << 3
};
} // namespace tf::intersect
