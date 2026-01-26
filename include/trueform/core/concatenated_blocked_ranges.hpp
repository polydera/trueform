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
#include "./algorithm/parallel_copy.hpp"
#include "./algorithm/parallel_copy_blocked.hpp"
#include "./apply.hpp"
#include "./blocked_buffer.hpp"
#include "./direction.hpp"
#include "./offset_block_buffer.hpp"
#include "./views/slice.hpp"
#include "./views/slide_range.hpp"

namespace tf {

/// @ingroup core_ranges
/// @brief Concatenate multiple blocked ranges into a single buffer.
///
/// Copies blocks from all input ranges into one output buffer.
/// If all ranges have the same static block size, returns a blocked_buffer.
/// Otherwise returns an offset_block_buffer for variable-length blocks.
///
/// @tparam Index The index type.
/// @tparam Range0 The first range type.
/// @tparam Range1 The second range type.
/// @tparam Ranges Additional range types.
/// @param r0 The first input range.
/// @param r1 The second input range.
/// @param rs Additional input ranges.
/// @return A blocked_buffer or offset_block_buffer containing all blocks.
template <typename Index, typename Range0, typename Range1, typename... Ranges>
auto concatenated_blocked_ranges(const Range0 &r0, const Range1 &r1,
                                 const Ranges &...rs) {
  // same fixed arity across ALL ranges and not dynamic?
  using all_same_static_size_t = std::integral_constant<
      bool, (tf::static_size_v<decltype(r0[0])> != tf::dynamic_size) &&
                (tf::static_size_v<decltype(r0[0])> ==
                 tf::static_size_v<decltype(r1[0])>) &&
                (true && ... &&
                 (tf::static_size_v<decltype(r0[0])> ==
                  tf::static_size_v<decltype(rs[0])>))>;
  constexpr bool all_same_static_size = all_same_static_size_t::value;

  const Index total_blocks =
      static_cast<Index>(r0.size() + r1.size() + (0 + ... + rs.size()));

  // Copy helper: writes each input range into [start_i, end_i) of 'out'
  auto run_copy = [&](auto &out) {
    Index start_i = 0;
    tf::apply(
        [&](const auto &...all) {
          auto copy_one = [&](const auto &r) {
            const Index end_i = static_cast<Index>(start_i + r.size());
            if constexpr (all_same_static_size_t::value) {
              tf::parallel_copy(r, tf::slice(out, start_i, end_i));
            } else {
              tf::parallel_copy_blocked(r, tf::slice(out, start_i, end_i));
            }
            start_i = end_i;
          };
          (copy_one(all), ...);
        },
        std::forward_as_tuple(r0, r1, rs...));
  };

  if constexpr (all_same_static_size) {
    // ---- Same fixed arity -> blocked buffer ----
    constexpr auto K = tf::static_size_v<decltype(r0[0])>;
    tf::blocked_buffer<Index, K> out;
    out.allocate(total_blocks);
    run_copy(out);
    return out;
  } else {
    // ---- Mixed/different arity (or dynamic) -> offset buffer ----
    tf::offset_block_buffer<Index, Index> out;
    auto &offsets = out.offsets_buffer();

    offsets.allocate(static_cast<Index>(total_blocks + 1));
    offsets[0] = 0;

    // Fill offsets for all input ranges sequentially
    Index start_f = 0;
    auto fill_offsets = [&](const auto &in_r) {
      const Index end_f = static_cast<Index>(start_f + in_r.size());
      auto slide = tf::slice(tf::make_slide_range<2>(offsets), start_f, end_f);
      for (auto &&[ofs, block] : tf::zip(slide, in_r)) {
        ofs[1] = ofs[0] + static_cast<Index>(block.size());
      }
      start_f = end_f;
    };
    tf::apply([&](const auto &...all) { (fill_offsets(all), ...); },
              std::forward_as_tuple(r0, r1, rs...));

    // Allocate payload and copy
    out.data_buffer().allocate(offsets.back());
    run_copy(out);
    return out;
  }
}

namespace core {
template <typename Index, typename Iterator0, std::size_t N0,
          typename Iterator1, std::size_t N1, typename... Iterators,
          std::size_t... Ns>
auto concatenated_blocked_ranges_directed(
    std::pair<tf::range<Iterator0, N0>, tf::direction> r0,
    std::pair<tf::range<Iterator1, N1>, tf::direction> r1,
    std::pair<tf::range<Iterators, Ns>, tf::direction>... rs) {
  // same fixed arity across ALL ranges and not dynamic?
  using all_same_static_size_t = std::integral_constant<
      bool, (tf::static_size_v<decltype(r0.first[0])> != tf::dynamic_size) &&
                (tf::static_size_v<decltype(r0.first[0])> ==
                 tf::static_size_v<decltype(r1.first[0])>) &&
                (true && ... &&
                 (tf::static_size_v<decltype(r0.first[0])> ==
                  tf::static_size_v<decltype(rs.first[0])>))>;
  constexpr bool all_same_static_size = all_same_static_size_t::value;

  const Index total_blocks = static_cast<Index>(
      r0.first.size() + r1.first.size() + (0 + ... + rs.first.size()));

  // Copy helper: writes each input range into [start_i, end_i) of 'out'
  auto run_copy = [&](auto &out) {
    Index start_i = 0;
    tf::apply(
        [&](const auto &...all) {
          auto copy_one = [&](const auto &p) {
            const auto &rng = p.first;
            const auto dir = p.second;

            const Index end_i = static_cast<Index>(start_i + rng.size());
            if constexpr (all_same_static_size_t::value) {
              if (dir == tf::direction::forward) {
                tf::parallel_copy(rng, tf::slice(out, start_i, end_i));
              } else {
                tf::parallel_copy_blocked_reverse(
                    rng, tf::slice(out, start_i, end_i));
              }
            } else {
              if (dir == tf::direction::forward) {
                tf::parallel_copy_blocked(rng, tf::slice(out, start_i, end_i));
              } else {
                tf::parallel_copy_blocked_reverse(
                    rng, tf::slice(out, start_i, end_i));
              }
            }
            start_i = end_i;
          };
          (copy_one(all), ...);
        },
        std::forward_as_tuple(r0, r1, rs...));
  };

  if constexpr (all_same_static_size) {
    // ---- Same fixed arity -> blocked buffer ----
    constexpr auto K = tf::static_size_v<decltype(r0.first[0])>;
    tf::blocked_buffer<Index, K> out;
    out.allocate(total_blocks);
    run_copy(out);
    return out;
  } else {
    // ---- Mixed/different arity (or dynamic) -> offset buffer ----
    tf::offset_block_buffer<Index, Index> out;
    auto &offsets = out.offsets_buffer();

    offsets.allocate(static_cast<Index>(total_blocks + 1));
    offsets[0] = 0;

    // Fill offsets for all input ranges sequentially
    Index start_f = 0;
    auto fill_offsets = [&](const auto &in_p) {
      const auto &in_r = in_p.first;
      const Index end_f = static_cast<Index>(start_f + in_r.size());
      auto slide = tf::slice(tf::make_slide_range<2>(offsets), start_f, end_f);
      for (auto &&[ofs, block] : tf::zip(slide, in_r)) {
        ofs[1] = ofs[0] + static_cast<Index>(block.size());
      }
      start_f = end_f;
    };
    tf::apply([&](const auto &...all) { (fill_offsets(all), ...); },
              std::forward_as_tuple(r0, r1, rs...));

    // Allocate payload and copy
    out.data_buffer().allocate(offsets.back());
    run_copy(out);

    return out;
  }
}
} // namespace core
} // namespace tf
