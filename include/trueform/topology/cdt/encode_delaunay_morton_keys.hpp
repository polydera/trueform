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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./delaunay_site_domain.hpp"
#include "./delaunay_site_partition_summary.hpp"
#include "./morton_code.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt {

template <bool Parallel, typename Owner, typename Int>
auto encode_delaunay_morton_keys(
    Owner &owner, const delaunay_site_domain<Int> &domain,
    std::size_t partition_count,
    tf::buffer<delaunay_site_partition_summary<Int>> &summaries)
    -> std::uint32_t {
  const std::size_t n_sites = owner._sites.size();
  auto remaining_high_bits = std::max(domain.span_x, domain.span_y) >> 16U;
  unsigned coordinate_shift = 0;
  while (remaining_high_bits != 0) {
    ++coordinate_shift;
    remaining_high_bits >>= 1U;
  }

  owner._keys.allocate(n_sites);
  owner._site_scratch.allocate(n_sites);
  owner._key_scratch.allocate(n_sites);
  std::uint32_t largest_key = 0;
  if constexpr (Parallel) {
    tf::parallel_for_each(
        tf::make_sequence_range(partition_count), [&](std::size_t partition) {
          const std::size_t first = n_sites * partition / partition_count;
          const std::size_t last = n_sites * (partition + 1) / partition_count;
          std::uint32_t partition_largest_key = 0;
          for (std::size_t i = first; i < last; ++i) {
            const auto x = static_cast<std::uint32_t>(
                domain.x_offset(owner._sites[i].x) >> coordinate_shift);
            const auto y = static_cast<std::uint32_t>(
                domain.y_offset(owner._sites[i].y) >> coordinate_shift);
            const auto key = morton_code(x, y);
            owner._keys[i] = key;
            partition_largest_key = std::max(partition_largest_key, key);
          }
          summaries[partition].largest_morton_key = partition_largest_key;
        });
    for (const auto &summary : summaries)
      largest_key = std::max(largest_key, summary.largest_morton_key);
  } else {
    for (std::size_t i = 0; i < n_sites; ++i) {
      const auto x = static_cast<std::uint32_t>(
          domain.x_offset(owner._sites[i].x) >> coordinate_shift);
      const auto y = static_cast<std::uint32_t>(
          domain.y_offset(owner._sites[i].y) >> coordinate_shift);
      const auto key = morton_code(x, y);
      owner._keys[i] = key;
      largest_key = std::max(largest_key, key);
    }
  }
  return largest_key;
}

} // namespace tf::topology::cdt
