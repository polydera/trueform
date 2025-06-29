/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../iter/zip_iterator.hpp"
#include "../range.hpp"

namespace tf {
template <typename Range> auto zip_ranges(Range &&r) -> Range && {
  return static_cast<Range &&>(r);
}
template <typename Range0, typename Range1, typename... Ranges>
auto zip_ranges(Range0 &&r0, Range1 &&r1, Ranges &&...r) {
  return tf::make_range(
      iter::make_zip_iterator(r0.begin(), r1.begin(), r.begin()...),
      iter::make_zip_iterator(r0.end(), r1.end(), r.end()...));
}
} // namespace tf
