/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
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
