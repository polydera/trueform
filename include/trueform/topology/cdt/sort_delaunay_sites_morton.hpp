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
#include "./certify_delaunay_site_predicates.hpp"
#include "./delaunay_execution_tuning.hpp"
#include "./encode_delaunay_morton_keys.hpp"
#include "./measure_delaunay_site_domain.hpp"
#include "./stable_radix_order_delaunay_sites.hpp"
#include <algorithm>
#include <cstddef>
#include <tbb/task_arena.h>

namespace tf::topology::cdt {

template <bool Parallel, typename Owner>
auto sort_delaunay_sites_morton_impl(Owner &owner) -> void {
  const std::size_t n_sites = owner._sites.size();
  std::size_t partition_count = 1;
  if constexpr (Parallel) {
    const std::size_t concurrency = std::max(
        std::size_t(1), std::size_t(tbb::this_task_arena::max_concurrency()));
    if (n_sites < delaunay_execution_tuning::parallel_order_sites ||
        concurrency < 2) {
      sort_delaunay_sites_morton_impl<false>(owner);
      return;
    }
    partition_count = std::min(
        concurrency,
        (n_sites + delaunay_execution_tuning::parallel_task_sites - 1) /
            delaunay_execution_tuning::parallel_task_sites);
  }

  const auto domain = measure_delaunay_site_domain<Parallel>(
      owner, partition_count, owner._partition_summaries);
  certify_delaunay_site_predicates(owner, domain);
  const auto largest_key = encode_delaunay_morton_keys<Parallel>(
      owner, domain, partition_count, owner._partition_summaries);
  stable_radix_order_delaunay_sites<Parallel>(owner, largest_key,
                                              partition_count);
}

template <typename Owner>
auto sort_delaunay_sites_morton(Owner &owner) -> void {
  sort_delaunay_sites_morton_impl<false>(owner);
}

template <typename Owner>
auto sort_delaunay_sites_morton_parallel(Owner &owner) -> void {
  sort_delaunay_sites_morton_impl<true>(owner);
}

} // namespace tf::topology::cdt
