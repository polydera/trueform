/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <functional>
namespace tf {
template <typename Range0, typename OutIter, typename Val, typename Compare>
auto compute_offsets(const Range0 &data, OutIter out_iter, Val initial_offset,
                     Compare &&compare) {
  if (!data.size())
    return out_iter;
  *out_iter++ = initial_offset++;
  auto iter = data.begin();
  auto end = data.end();
  auto current = iter;
  while (++iter != end) {
    if (!compare(*current, *iter)) {
      *out_iter++ = initial_offset;
      current = iter;
    }
    ++initial_offset;
  }
  *out_iter++ = initial_offset;
  return out_iter;
}

template <typename Range0, typename OutIter, typename Val>
auto compute_offsets(const Range0 &data, OutIter out_iter, Val initial_offset) {
  return compute_offsets(data, out_iter, initial_offset, std::equal_to<>{});
}
} // namespace tf
