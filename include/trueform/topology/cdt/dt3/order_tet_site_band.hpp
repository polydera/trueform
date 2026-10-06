/*
 * Copyright (c) 2026 XLAB
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
#include "./tet_execution_tuning.hpp"
#include "./tet_site_key.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Morton-order one rank band; equal codes retain canonical-name rank.
template <typename Owner>
auto order_tet_site_band(Owner &owner, std::size_t first, std::size_t last)
    -> void {
  using Index = typename Owner::index_type;
  const unsigned radix_bits = tet_execution_tuning::order_radix_bits;
  const std::size_t radix_size = std::size_t(1) << radix_bits;
  const std::uint64_t radix_mask = std::uint64_t(radix_size - 1);

  const std::size_t n = last - first;
  owner._key_records.allocate(n);
  std::uint64_t largest = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Index site = owner._order[first + i];
    const auto code = owner._keys[std::size_t(site)];
    owner._key_records[i] = {code, site};
    largest = std::max(largest, code);
  }

  if (n < tet_execution_tuning::comparison_order_sites) {
    const auto *priorities = owner._priorities.data();
    const auto *sites = owner._sites.data();
    std::sort(owner._key_records.begin(), owner._key_records.end(),
              [priorities, sites](const tet_site_key<Index> &x,
                                  const tet_site_key<Index> &y) {
                if (x.code != y.code)
                  return x.code < y.code;
                const auto a = std::size_t(x.site), b = std::size_t(y.site);
                return priorities[a] < priorities[b] ||
                       (priorities[a] == priorities[b] &&
                        sites[a].id < sites[b].id);
              });
  } else {
    std::size_t pass_count = 0;
    do {
      ++pass_count;
      largest >>= radix_bits;
    } while (largest != 0);

    owner._key_scratch.allocate(n);
    owner._counts.allocate(radix_size);
    for (std::size_t pass = 0; pass < pass_count; ++pass) {
      std::fill(owner._counts.begin(), owner._counts.end(), std::size_t(0));
      const auto shift = unsigned(pass * radix_bits);
      for (const auto &record : owner._key_records)
        ++owner._counts[std::size_t((record.code >> shift) & radix_mask)];

      std::size_t next_output = 0;
      for (auto &output : owner._counts) {
        const std::size_t after_digit = next_output + output;
        output = next_output;
        next_output = after_digit;
      }
      for (const auto &record : owner._key_records) {
        const auto digit = std::size_t((record.code >> shift) & radix_mask);
        owner._key_scratch[owner._counts[digit]++] = record;
      }
      std::swap(owner._key_records, owner._key_scratch);
    }
  }

  for (std::size_t i = 0; i < n; ++i)
    owner._order[first + i] = owner._key_records[i].site;
}

} // namespace tf::topology::cdt::dt3
