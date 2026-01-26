/*
* Copyright (c) 2025 XLAB
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
#include "../buffer.hpp"
#include "./parallel_fill.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Build equivalence class map from identified pairs (dense).
///
/// Uses union-find to group elements by equivalence pairs.
/// All elements get mapped, even if not in any pair.
///
/// @tparam PairRange Range of (a, b) pairs.
/// @tparam MapRange Output map range type.
/// @param identified_pairs Pairs of equivalent element IDs.
/// @param map Output map from element ID to class ID.
/// @return The number of equivalence classes.
template <typename PairRange, typename MapRange>
auto make_dense_equivalence_class_map(const PairRange &identified_pairs,
                                      MapRange &map) {
  using Index = std::decay_t<decltype(map[0])>;
  const Index none = static_cast<Index>(map.size());

  tf::buffer<Index> root_map;
  root_map.allocate(map.size());
  tf::parallel_fill(root_map, none);
  tf::parallel_fill(map, none);

  auto find = [&](Index x) -> Index {
    if (root_map[x] == none) {
      root_map[x] = x;
      return x;
    }
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

  for (Index i = 0; i < static_cast<Index>(map.size()); ++i) {
    if (root_map[i] == none) {
      map[i] = current_id++;
    } else {
      Index root = find(i);
      if (root_to_id[root] == none) {
        root_to_id[root] = current_id++;
      }
      map[i] = root_to_id[root];
    }
  }
  return current_id;
}

/// @ingroup core_algorithms
/// @brief Build equivalence class map from identified pairs (sparse).
///
/// Only elements appearing in pairs get mapped.
/// Elements not in any pair remain unmapped (none tag).
///
/// @tparam PairRange Range of (a, b) pairs.
/// @tparam MapRange Output map range type.
/// @param identified_pairs Pairs of equivalent element IDs.
/// @param map Output map from element ID to class ID.
/// @return The number of equivalence classes.
template <typename PairRange, typename MapRange>
auto make_sparse_equivalence_class_map(const PairRange &identified_pairs,
                                       MapRange &map) {
  using Index = std::decay_t<decltype(map[0])>;
  const Index none = static_cast<Index>(map.size());

  tf::buffer<Index> root_map;
  root_map.allocate(map.size());
  tf::parallel_fill(root_map, none);
  tf::parallel_fill(map, none);

  auto find = [&](Index x) -> Index {
    if (root_map[x] == none) {
      root_map[x] = x;
      return x;
    }
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

  for (Index i = 0; i < static_cast<Index>(map.size()); ++i) {
    if (root_map[i] != none) {
      Index root = find(i);
      if (root_to_id[root] == none) {
        root_to_id[root] = current_id++;
      }
      map[i] = root_to_id[root];
    }
  }
  return current_id;
}

} // namespace tf
