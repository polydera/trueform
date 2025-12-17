/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <type_traits>

namespace tf {
template <typename Range0, typename F, typename Range1, typename Index>
auto remove_if_and_make_map(Range0 &data, const F &predicate, Range1 &map,
                            Index none_tag) {
  Index current_id = 0;
  auto it0 = data.begin();
  auto write_to = it0;
  auto end0 = data.end();
  auto it1 = map.begin();
  for (; it0 != end0; ++it0, ++it1) {
    if (!predicate(*it0)) {
      *it1 = current_id++;
      *write_to++ = *it0;
    } else
      *it1 = none_tag;
  }
  return write_to;
}

template <typename Range0, typename F, typename Range1>
auto remove_if_and_make_map(Range0 &data, const F &predicate, Range1 &map) {
  using Index = std::decay_t<decltype(map[0])>;
  return remove_if_and_make_map(data, predicate, map, Index(data.size()));
}
} // namespace tf
