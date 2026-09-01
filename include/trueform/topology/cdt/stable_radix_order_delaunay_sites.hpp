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
#include "../../core/views/sequence_range.hpp"
#include "./delaunay_execution_tuning.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::topology::cdt {

template <bool Parallel, typename Owner>
auto stable_radix_order_delaunay_sites(Owner &owner, std::uint32_t largest_key,
                                       std::size_t partition_count) -> void {
  constexpr unsigned radix_bits = 11;
  const std::size_t radix_size = std::size_t(1) << radix_bits;
  const std::uint32_t radix_mask = std::uint32_t(radix_size - 1);
  std::size_t pass_count = 0;
  do {
    ++pass_count;
    largest_key >>= radix_bits;
  } while (largest_key != 0);

  const std::size_t n_sites = owner._sites.size();
  // Small inputs never amortize the per-pass bucket table: order by one
  // comparison sort over (key, input index) — the index tiebreak IS the
  // radix's stability, so the permutation is identical bit for bit.
  if constexpr (sizeof(std::size_t) >= 8) {
    if (n_sites < delaunay_execution_tuning::comparison_order_sites) {
      owner._counts.allocate(n_sites);
      for (std::size_t i = 0; i < n_sites; ++i)
        owner._counts[i] =
            (std::size_t(owner._keys[i]) << 32) | std::size_t(i);
      std::sort(owner._counts.begin(), owner._counts.end());
      for (std::size_t i = 0; i < n_sites; ++i) {
        const auto at = std::size_t(owner._counts[i] & 0xffffffffull);
        owner._site_scratch[i] = owner._sites[at];
        owner._key_scratch[i] = owner._keys[at];
      }
      std::swap(owner._sites, owner._site_scratch);
      std::swap(owner._keys, owner._key_scratch);
      return;
    }
  }
  if constexpr (Parallel) {
    owner._counts.allocate(partition_count * radix_size);
    for (std::size_t pass = 0; pass < pass_count; ++pass) {
      const unsigned shift = static_cast<unsigned>(pass * radix_bits);
      tf::parallel_for_each(
          tf::make_sequence_range(partition_count), [&](std::size_t partition) {
            auto *counts = owner._counts.begin() + partition * radix_size;
            std::fill(counts, counts + radix_size, std::size_t(0));
            const std::size_t first = n_sites * partition / partition_count;
            const std::size_t last =
                n_sites * (partition + 1) / partition_count;
            for (std::size_t i = first; i < last; ++i) {
              const auto digit =
                  std::size_t((owner._keys[i] >> shift) & radix_mask);
              ++counts[digit];
            }
          });

      std::size_t next_output = 0;
      for (std::size_t digit = 0; digit < radix_size; ++digit) {
        for (std::size_t partition = 0; partition < partition_count;
             ++partition) {
          auto &partition_output =
              owner._counts[partition * radix_size + digit];
          const std::size_t after_partition = next_output + partition_output;
          partition_output = next_output;
          next_output = after_partition;
        }
      }

      tf::parallel_for_each(
          tf::make_sequence_range(partition_count), [&](std::size_t partition) {
            auto *outputs = owner._counts.begin() + partition * radix_size;
            const std::size_t first = n_sites * partition / partition_count;
            const std::size_t last =
                n_sites * (partition + 1) / partition_count;
            for (std::size_t i = first; i < last; ++i) {
              const auto key = owner._keys[i];
              const auto digit = std::size_t((key >> shift) & radix_mask);
              const std::size_t output = outputs[digit]++;
              owner._site_scratch[output] = owner._sites[i];
              owner._key_scratch[output] = key;
            }
          });
      std::swap(owner._sites, owner._site_scratch);
      std::swap(owner._keys, owner._key_scratch);
    }
  } else {
    owner._counts.allocate(radix_size);
    for (std::size_t pass = 0; pass < pass_count; ++pass) {
      std::fill(owner._counts.begin(), owner._counts.end(), std::size_t(0));
      const unsigned shift = static_cast<unsigned>(pass * radix_bits);
      for (const auto key : owner._keys) {
        const auto digit = std::size_t((key >> shift) & radix_mask);
        ++owner._counts[digit];
      }

      std::size_t next_output = 0;
      for (auto &output : owner._counts) {
        const std::size_t after_digit = next_output + output;
        output = next_output;
        next_output = after_digit;
      }
      for (std::size_t i = 0; i < n_sites; ++i) {
        const auto key = owner._keys[i];
        const auto digit = std::size_t((key >> shift) & radix_mask);
        const std::size_t output = owner._counts[digit]++;
        owner._site_scratch[output] = owner._sites[i];
        owner._key_scratch[output] = key;
      }
      std::swap(owner._sites, owner._site_scratch);
      std::swap(owner._keys, owner._key_scratch);
    }
  }
}

} // namespace tf::topology::cdt
