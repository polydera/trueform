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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include <algorithm>
#include <numeric>

#include <tbb/parallel_sort.h>

namespace tf {

// Size-adaptive primitives: callers routinely build thousands of tiny
// triangulations from inside parallel workers -- parallel dispatch on a
// handful of elements costs more than the work, so everything below the
// cutoff runs serial with zero task overhead.
namespace topology::detail {
constexpr std::size_t k_serial_cutoff = 2048;
template <typename Buf, typename T>
inline auto fill_auto(Buf &b, const T &v) -> void {
  if (std::size_t(b.size()) < k_serial_cutoff)
    std::fill(b.begin(), b.end(), v);
  else
    tf::parallel_fill(b, v);
}
template <typename Buf, typename T>
inline auto iota_auto(Buf &b, T v0) -> void {
  if (std::size_t(b.size()) < k_serial_cutoff)
    std::iota(b.begin(), b.end(), v0);
  else
    tf::parallel_iota(b, v0);
}
template <typename It, typename... Cmp>
inline auto sort_auto(It b, It e, Cmp... cmp) -> void {
  if (std::size_t(e - b) < k_serial_cutoff)
    std::sort(b, e, cmp...);
  else
    tbb::parallel_sort(b, e, cmp...);
}

} // namespace topology::detail

} // namespace tf
