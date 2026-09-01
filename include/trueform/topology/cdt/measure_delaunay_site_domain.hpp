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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::topology::cdt {

template <bool Parallel, typename Owner>
auto measure_delaunay_site_domain(
    const Owner &owner, std::size_t partition_count,
    tf::buffer<delaunay_site_partition_summary<typename Owner::int_type>>
        &summaries) -> delaunay_site_domain<typename Owner::int_type> {
  using Int = typename Owner::int_type;
  using UInt = std::make_unsigned_t<Int>;
  const std::size_t n_sites = owner._sites.size();
  Int minimum_x = owner._sites[0].x;
  Int minimum_y = owner._sites[0].y;
  Int maximum_x = minimum_x;
  Int maximum_y = minimum_y;

  if constexpr (Parallel) {
    summaries.allocate(partition_count);
    tf::parallel_for_each(
        tf::make_sequence_range(partition_count), [&](std::size_t partition) {
          const std::size_t first = n_sites * partition / partition_count;
          const std::size_t last = n_sites * (partition + 1) / partition_count;
          const auto &first_site = owner._sites[first];
          auto summary = delaunay_site_partition_summary<Int>{
              first_site.x, first_site.y, first_site.x, first_site.y,
              std::uint32_t(0)};
          for (std::size_t i = first + 1; i < last; ++i) {
            summary.minimum_x = std::min(summary.minimum_x, owner._sites[i].x);
            summary.minimum_y = std::min(summary.minimum_y, owner._sites[i].y);
            summary.maximum_x = std::max(summary.maximum_x, owner._sites[i].x);
            summary.maximum_y = std::max(summary.maximum_y, owner._sites[i].y);
          }
          summaries[partition] = summary;
        });
    for (const auto &summary : summaries) {
      minimum_x = std::min(minimum_x, summary.minimum_x);
      minimum_y = std::min(minimum_y, summary.minimum_y);
      maximum_x = std::max(maximum_x, summary.maximum_x);
      maximum_y = std::max(maximum_y, summary.maximum_y);
    }
  } else {
    for (std::size_t i = 1; i < n_sites; ++i) {
      minimum_x = std::min(minimum_x, owner._sites[i].x);
      minimum_y = std::min(minimum_y, owner._sites[i].y);
      maximum_x = std::max(maximum_x, owner._sites[i].x);
      maximum_y = std::max(maximum_y, owner._sites[i].y);
    }
  }

  return {minimum_x, minimum_y,
          std::uint64_t(UInt(maximum_x) - UInt(minimum_x)),
          std::uint64_t(UInt(maximum_y) - UInt(minimum_y))};
}

} // namespace tf::topology::cdt
