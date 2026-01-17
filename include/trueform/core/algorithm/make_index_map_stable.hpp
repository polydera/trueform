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
#include "../views/enumerate.hpp"
#include "./parallel_for_each.hpp"
#include "./parallel_iota.hpp"
#include "tbb/parallel_sort.h"
namespace tf {

/// @ingroup core_algorithms
/// @brief Reorder index map to preserve original element order.
///
/// Sorts kept_ids by their original indices so the output
/// maintains stable ordering relative to the input.
///
/// @tparam Index The index type.
/// @param src Index map to stabilize (modified in-place).
/// @param none_tag Value indicating unmapped elements.
template <typename Index>
auto make_index_map_stable(tf::index_map_buffer<Index> &src, Index none_tag) {
  // [b, c, a] -> [a, b, c]
  // kept_ids_prime: [2, 0, 1]
  // f_prime:        [1, 2, 0]
  tf::buffer<Index> kept_ids_prime;
  kept_ids_prime.allocate(src.kept_ids().size());
  tf::parallel_iota(kept_ids_prime, 0);
  tbb::parallel_sort(tf::zip(src.kept_ids(), kept_ids_prime));
  tf::buffer<Index> f_prime;
  f_prime.allocate(src.kept_ids().size());
  for (auto [i, e] : tf::enumerate(kept_ids_prime)) {
    f_prime[e] = i;
  }
  tf::parallel_for_each(
      src.f(),
      [&](auto &x) {
        if (x != none_tag)
          x = f_prime[x];
      },
      tf::checked);
}

} // namespace tf
