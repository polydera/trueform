/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../iter/zip_iterator.hpp"
#include "../range.hpp"

namespace tf {
template <typename Range0, typename Range1, typename... Ranges>
auto zip(Range0 &&r0, Range1 &&r1, Ranges &&...r) {
  return tf::make_range(
      iter::make_zip_iterator(r0.begin(), r1.begin(), r.begin()...),
      iter::make_zip_iterator(r0.end(), r1.end(), r.end()...));
}
} // namespace tf
