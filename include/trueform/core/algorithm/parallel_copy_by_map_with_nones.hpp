/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../views/zip.hpp"
#include "./parallel_apply.hpp"

namespace tf {
template <typename Range0, typename Range1, typename Range2, typename Index>
auto parallel_copy_by_map_with_nones(const Range0 &src, Range1 &&dst,
                                     const Range2 &map, Index none) {
  tf::parallel_apply(tf::zip(src, map), [&](auto pair) {
    auto &[_in, _id] = pair;
    if (_id != none)
      dst[_id] = _in;
  });
}

template <typename Range0, typename Range1, typename Range2>
auto parallel_copy_by_map_with_nones(const Range0 &src, Range1 &&dst,
                                     const Range2 &map) {
  using Index = std::decay_t<decltype(map[0])>;
  parallel_copy_by_map_with_nones(src, dst, map, Index(map.size()));
}
} // namespace tf
