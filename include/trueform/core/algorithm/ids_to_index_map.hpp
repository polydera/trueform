/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../index_map.hpp"
#include "../views/indirect_range.hpp"
#include "../views/sequence_range.hpp"
#include "./parallel_copy.hpp"
#include "./parallel_fill.hpp"

namespace tf {

template <typename Index, typename Range>
auto ids_to_index_map(const Range &ids, tf::index_map_buffer<Index> &mapping,
                      Index total_elements) {
  mapping.f().allocate(total_elements);
  tf::parallel_fill(mapping.f(), total_elements);

  mapping.kept_ids().allocate(ids.size());
  tf::parallel_copy(ids, mapping.kept_ids());

  tf::parallel_copy(tf::make_sequence_range(mapping.kept_ids().size()),
                    tf::make_indirect_range(mapping.kept_ids(), mapping.f()));
}

template <typename Index, typename Range>
auto ids_to_index_map(const Range &ids, Index total_elements) {
  tf::index_map_buffer<Index> mapping;
  tf::ids_to_index_map<Index>(ids, mapping, total_elements);
  return mapping;
}

} // namespace tf
