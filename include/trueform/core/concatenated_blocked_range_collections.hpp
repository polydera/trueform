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
#include "./blocked_buffer.hpp"
#include "./offset_block_buffer.hpp"
#include "./views/slice.hpp"
#include "./views/slide_range.hpp"
#include <type_traits>

namespace tf {

/// @ingroup core_ranges
/// @brief Concatenate collections of blocked ranges into a single buffer.
///
/// Flattens multiple collections of sub-ranges into one output buffer.
/// Each collection contains sub-ranges that are themselves blocked ranges.
/// Optimizes for same-arity blocks using blocked_buffer, otherwise uses
/// offset_block_buffer.
///
/// @tparam Index The index type.
/// @tparam Range0 The first collection type.
/// @tparam Range1 The second collection type.
/// @tparam Ranges Additional collection types.
/// @param r0 The first collection of sub-ranges.
/// @param r1 The second collection of sub-ranges.
/// @param rs Additional collections.
/// @return A blocked_buffer or offset_block_buffer containing all blocks.
template <typename Index, typename Range0, typename Range1, typename... Ranges>
auto concatenated_blocked_range_collections(const Range0 &r0, const Range1 &r1,
                                            const Ranges &...rs) {
  // Deduce static block arity of a collection's sub-range element type.
  constexpr std::size_t K0 = tf::static_size_v<decltype(r0.front().front())>;
  constexpr std::size_t K1 = tf::static_size_v<decltype(r1.front().front())>;

  // Are ALL collections composed of sub-ranges with the same fixed
  // (non-dynamic) arity?
  constexpr bool all_same_static_size =
      (K0 != tf::dynamic_size) && (K0 == K1) &&
      (true && ... && (K0 == tf::static_size_v<decltype(rs.front().front())>));

  // Total number of blocks = sum of sizes of all sub-ranges in all collections
  auto sum_blocks = [](const auto &coll) -> Index {
    Index s = 0;
    for (const auto &sub : coll)
      s += static_cast<Index>(sub.size());
    return s;
  };
  const Index total_blocks =
      sum_blocks(r0) + sum_blocks(r1) + (Index{0} + ... + sum_blocks(rs));

  // Helper to iterate all collections with a callable f(collection)
  auto for_each_collection = [&](auto &&f) {
    f(r0);
    f(r1);
    (f(rs), ...);
  };

  if constexpr (all_same_static_size) {
    // ---- Same fixed arity across all sub-ranges → blocked buffer ----
    constexpr auto K = K0;
    tf::blocked_buffer<Index, K> out;
    out.allocate(total_blocks);

    Index start = 0;
    auto copy_sub = [&](const auto &sub) {
      const Index end = start + static_cast<Index>(sub.size());
      tf::parallel_copy(sub, tf::slice(out, start, end));
      start = end;
    };
    for_each_collection([&](const auto &coll) {
      for (const auto &sub : coll)
        copy_sub(sub);
    });

    return out;

  } else {
    // ---- Mixed/different (or dynamic) arities → offset buffer ----
    tf::offset_block_buffer<Index, Index> out;
    auto &offsets = out.offsets_buffer();

    offsets.allocate(total_blocks + Index{1});
    offsets[0] = 0;

    // Fill offsets by walking each sub-range. Fixed-arity uses arithmetic
    // stride, dynamic arity uses local prefix-sum of block sizes.
    Index start_i = 0;

    auto fill_offsets_for_sub = [&](const auto &sub) {
      using SubRange = std::decay_t<decltype(sub)>;
      using Elem =
          std::decay_t<decltype(*std::begin(std::declval<SubRange &>()))>;
      constexpr std::size_t Ksub = tf::static_size_v<Elem>;

      const Index end_i = start_i + static_cast<Index>(sub.size());

      if constexpr (Ksub != tf::dynamic_size) {
        // Write offsets[start_i .. end_i] and terminal at end_i as an
        // arithmetic progression.
        auto seg = tf::slice(offsets, start_i, end_i + Index{1});
        Index base = seg[0];
        for (Index i = 0; i < static_cast<Index>(seg.size()); ++i)
          seg[i] = base + static_cast<Index>(Ksub) * i;
      } else {
        // Dynamic: prefix-sum block extents into offsets[start_i .. end_i]
        auto slide =
            tf::slice(tf::make_slide_range<2>(offsets), start_i, end_i);
        for (auto &&[ofs, block] : tf::zip(slide, sub))
          ofs[1] = ofs[0] + static_cast<Index>(block.size());
      }

      start_i = end_i;
    };

    for_each_collection([&](const auto &coll) {
      for (const auto &sub : coll)
        fill_offsets_for_sub(sub);
    });

    // Allocate payload and copy the blocks
    out.data_buffer().allocate(offsets.back());

    Index start = 0;
    auto copy_sub = [&](const auto &sub) {
      const Index end = start + static_cast<Index>(sub.size());
      tf::parallel_copy_blocked(sub, tf::slice(out, start, end));
      start = end;
    };
    for_each_collection([&](const auto &coll) {
      for (const auto &sub : coll)
        copy_sub(sub);
    });

    return out;
  }
}

} // namespace tf
