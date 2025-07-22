/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../buffer.hpp"
#include "./parallel_fill.hpp"

namespace tf {

template <typename PairRange, typename MapRange>
auto make_equivalence_class_map(const PairRange &identified_pairs,
                                MapRange &map) {
  using Index = std::decay_t<decltype(map[0])>;
  const Index none = static_cast<Index>(map.size());

  tf::parallel_fill(map, none);
  for (const auto &[a, b] : identified_pairs) {
    map[a] = a;
    map[b] = b;
  }

  auto find = [&](Index x) -> Index {
    Index root = x;
    while (map[root] != root)
      root = map[root];
    while (map[x] != root) {
      Index parent = map[x];
      map[x] = root;
      x = parent;
    }
    return root;
  };

  for (const auto &[a, b] : identified_pairs) {
    Index ra = find(a);
    Index rb = find(b);
    if (ra != rb)
      map[rb] = ra;
  }

  // Assign compact IDs to roots
  tf::buffer<Index> root_to_id;
  root_to_id.allocate(map.size());
  tf::parallel_fill(root_to_id, none);
  Index current_id = 0;

  for (Index i = 0; i < static_cast<Index>(map.size()); ++i) {
    if (map[i] == none) {
      map[i] = current_id++;
    } else {
      Index root = find(i);
      if (root_to_id[root] == none)
        root_to_id[root] = current_id++;
      map[i] = root_to_id[root];
    }
  }
  return current_id;
}
} // namespace tf
