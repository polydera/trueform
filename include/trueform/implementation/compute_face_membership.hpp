/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../parallel_fill.hpp"

namespace tf::implementation {
template <typename Range0, typename Range1, typename Range2, typename F>
auto compute_face_membership(const Range0 &blocked_range, Range1 &&offsets,
                        Range2 &&links, const F &make_link_element) {

  auto number_of_used_ids = links.size();
  auto number_of_unique_ids = offsets.size() - 1;

  offsets[number_of_unique_ids] = number_of_used_ids;
  tf::parallel_fill(offsets, 0);

  for (const auto &range : blocked_range) {
    for (const auto &id : range)
      offsets[id]++;
  }

  for (decltype(number_of_unique_ids) point_id = 0;
       point_id < number_of_unique_ids; ++point_id) {
    offsets[point_id + 1] += offsets[point_id];
  }

  std::size_t block_id = 0;
  for (const auto &range : blocked_range) {
    std::size_t sub_id = 0;
    for (const auto &id : range) {
      offsets[id]--;
      links[offsets[id]] = make_link_element(sub_id++, block_id);
    }
    ++block_id;
  }
  offsets[number_of_unique_ids] = number_of_used_ids;
}

template <typename Range0, typename Range1, typename Range2>
auto compute_face_membership(const Range0 &blocked_range, Range1 &&offsets,
                        Range2 &&links) {
  compute_face_membership(blocked_range, offsets, links,
                     [](auto, auto block_id) { return block_id; });
}
} // namespace tf::implementation
