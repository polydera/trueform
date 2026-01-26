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
#include "../index_map.hpp"
#include "./parallel_fill.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Build equivalence class index map (dense).
///
/// Creates an index map where equivalent elements map to the same ID.
/// The representative of each class is the smallest ID in that class.
///
/// @tparam PairRange Range of (a, b) pairs.
/// @tparam Index The index type.
/// @param identified_pairs Pairs of equivalent element IDs.
/// @param n_ids Total number of element IDs.
/// @param im Output index map buffer.
/// @return The number of equivalence classes.
template <typename PairRange, typename Index>
auto make_dense_equivalence_class_index_map(const PairRange &identified_pairs,
                                            std::size_t n_ids,
                                            tf::index_map_buffer<Index> &im) {
  im.f().allocate(n_ids);
  auto &map = im.f();
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
    if (rb < ra)
      std::swap(ra, rb);
    if (ra != rb)
      root_map[rb] = ra;
  }

  // Assign compact IDs to roots
  Index current_id = 0;
  im.kept_ids().reserve(n_ids);

  for (Index i = 0; i < static_cast<Index>(map.size()); ++i) {
    if (root_map[i] == none) {
      im.kept_ids().push_back(i);
      map[i] = current_id++;
    } else {
      Index root = find(i);
      // root is always the smallest
      // off the the is in the eq. class
      if (i == root) {
        im.kept_ids().push_back(i);
        map[root] = current_id++;
      } else {
        map[i] = map[root];
      }
    }
  }
  return current_id;
}

/// @ingroup core_algorithms
/// @brief Build and return equivalence class index map (dense).
/// @overload
template <typename Index, typename PairRange>
auto make_dense_equivalence_class_index_map(const PairRange &identified_pairs,
                                            std::size_t n_ids) {
  tf::index_map_buffer<Index> im;
  make_dense_equivalence_class_index_map(identified_pairs, n_ids, im);
  return im;
}

/// @ingroup core_algorithms
/// @brief Build equivalence class index map (sparse).
///
/// Only elements in pairs are mapped; others remain unmapped.
///
/// @tparam PairRange Range of (a, b) pairs.
/// @tparam Index The index type.
/// @param identified_pairs Pairs of equivalent element IDs.
/// @param n_ids Total number of element IDs.
/// @param im Output index map buffer.
/// @return The number of equivalence classes.
template <typename PairRange, typename Index>
auto make_sparse_equivalence_class_index_map(const PairRange &identified_pairs,
                                             std::size_t n_ids,
                                             tf::index_map_buffer<Index> &im) {
  im.f().allocate(n_ids);
  auto &map = im.f();
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
    if (rb < ra)
      std::swap(ra, rb);
    if (ra != rb)
      root_map[rb] = ra;
  }

  // Assign compact IDs to roots
  Index current_id = 0;
  im.kept_ids().reserve(n_ids);

  for (Index i = 0; i < static_cast<Index>(map.size()); ++i) {
    if (root_map[i] != none) {
      Index root = find(i);
      // root is always the smallest
      // off the the is in the eq. class
      if (i == root) {
        im.kept_ids().push_back(i);
        map[root] = current_id++;
      } else {
        map[i] = map[root];
      }
    }
  }
  return current_id;
}

/// @ingroup core_algorithms
/// @brief Build and return equivalence class index map (sparse).
/// @overload
template <typename Index, typename PairRange>
auto make_sparse_equivalence_class_index_map(const PairRange &identified_pairs,
                                             std::size_t n_ids) {
  tf::index_map_buffer<Index> im;
  make_sparse_equivalence_class_index_map(identified_pairs, n_ids, im);
  return im;
}
} // namespace tf
