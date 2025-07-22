/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../index_map.hpp"
#include "./parallel_apply.hpp"
#include "./remove_if_and_make_map.hpp"

namespace tf {
template <typename Index, typename Range0>
auto update_by_mask(tf::index_map_buffer<Index> &im, const Range0 &mask) {
  tf::buffer<Index> map;
  map.allocate(im.kept_ids().size());
  auto none = Index(mask.size());
  im.kept_ids().erase_till_end(tf::remove_if_and_make_map(
      im.kept_ids(), [&](Index id) { return !mask[id]; }, map, none));
  tf::parallel_apply(im.f(), [&](Index &id) {
    if (id != none)
      id = map[id];
  });
}
} // namespace tf
