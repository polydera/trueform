/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../buffer.hpp"
#include "../index_map.hpp"
#include "./parallel_fill.hpp"

namespace tf {

template <typename PairRange, typename Index>
auto make_equivalence_class_index_map(const PairRange &identified_pairs,
                                      std::size_t n_ids,
                                      tf::index_map_buffer<Index> &im) {
  im.f().allocate(n_ids);
  auto &map = im.f();
  const Index none = static_cast<Index>(map.size());
  tf::buffer<Index> root_map;
  root_map.allocate(map.size());
  tf::parallel_fill(root_map, none);

  tf::parallel_fill(map, none);
  for (const auto &[a, b] : identified_pairs) {
    root_map[a] = a;
    root_map[b] = b;
  }

  auto find = [&](Index x) -> Index {
    Index root = x;
    while (root_map[root] != root)
      root = root_map[root];
    while (root_map[x] != root) {
      Index parent = root_map[x];
      root_map[x] = root;
      x = parent;
    }
    return root;
  };

  for (const auto &[a, b] : identified_pairs) {
    Index ra = find(a);
    Index rb = find(b);
    if (ra != rb)
      root_map[rb] = ra;
  }

  // Assign compact IDs to roots
  tf::buffer<Index> root_to_id;
  root_to_id.allocate(map.size());
  tf::parallel_fill(root_to_id, none);
  Index current_id = 0;
  im.kept_ids().reserve(n_ids);

  for (Index i = 0; i < static_cast<Index>(map.size()); ++i) {
    if (root_map[i] == none) {
      im.kept_ids().push_back(i);
      map[i] = current_id++;
    } else {
      Index root = find(i);
      if (root_to_id[root] == none) {
        im.kept_ids().push_back(i);
        root_to_id[root] = current_id++;
      }
      map[i] = root_to_id[root];
    }
  }
  return current_id;
}

template <typename Index, typename PairRange>
auto make_equivalence_class_index_map(const PairRange &identified_pairs,
                                      std::size_t n_ids) {
  tf::index_map_buffer<Index> im;
  make_equivalence_class_index_map(identified_pairs, n_ids, im);
  return im;
}
} // namespace tf
