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

namespace core {

template <typename Index, typename Collection>
auto sum_blocks_in(const Collection &coll) -> Index {
  Index s = 0;
  for (const auto &sub : coll)
    s += static_cast<Index>(sub.size());
  return s;
}

template <typename Index, typename SubRange, typename Buffer>
auto copy_sub_blocked(const SubRange &sub, Buffer &out, Index &start) -> void {
  const Index end = start + static_cast<Index>(sub.size());
  tf::parallel_copy(sub, tf::slice(out, start, end));
  start = end;
}

template <typename Index, typename SubRange, typename Buffer>
auto copy_sub_offset(const SubRange &sub, Buffer &out, Index &start) -> void {
  const Index end = start + static_cast<Index>(sub.size());
  tf::parallel_copy_blocked(sub, tf::slice(out, start, end));
  start = end;
}

template <typename Index, typename SubRange>
auto fill_offsets_for_sub(const SubRange &sub, tf::buffer<Index> &offsets,
                          Index &start_i) -> void {
  using Elem =
      std::decay_t<decltype(*std::begin(std::declval<SubRange &>()))>;
  constexpr std::size_t Ksub = tf::static_size_v<Elem>;

  const Index end_i = start_i + static_cast<Index>(sub.size());

  if constexpr (Ksub != tf::dynamic_size) {
    auto seg = tf::slice(offsets, start_i, end_i + Index{1});
    Index base = seg[0];
    for (Index i = 0; i < static_cast<Index>(seg.size()); ++i)
      seg[i] = base + static_cast<Index>(Ksub) * i;
  } else {
    auto slide =
        tf::slice(tf::make_slide_range<2>(offsets), start_i, end_i);
    for (auto &&[ofs, block] : tf::zip(slide, sub))
      ofs[1] = ofs[0] + static_cast<Index>(block.size());
  }

  start_i = end_i;
}

} // namespace core

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
  constexpr std::size_t K0 = tf::static_size_v<decltype(r0.front().front())>;
  constexpr std::size_t K1 = tf::static_size_v<decltype(r1.front().front())>;

  constexpr bool all_same_static_size =
      (K0 != tf::dynamic_size) && (K0 == K1) &&
      (true && ... && (K0 == tf::static_size_v<decltype(rs.front().front())>));

  const Index total_blocks = core::sum_blocks_in<Index>(r0) +
                             core::sum_blocks_in<Index>(r1) +
                             (Index{0} + ... + core::sum_blocks_in<Index>(rs));

  if constexpr (all_same_static_size) {
    constexpr auto K = K0;
    tf::blocked_buffer<Index, K> out;
    out.allocate(total_blocks);

    Index start = 0;
    auto copy_coll = [&](const auto &coll) {
      for (const auto &sub : coll)
        core::copy_sub_blocked<Index>(sub, out, start);
    };
    copy_coll(r0);
    copy_coll(r1);
    (copy_coll(rs), ...);

    return out;

  } else {
    tf::offset_block_buffer<Index, Index> out;
    auto &offsets = out.offsets_buffer();

    offsets.allocate(total_blocks + Index{1});
    offsets[0] = 0;

    Index start_i = 0;
    auto fill_coll = [&](const auto &coll) {
      for (const auto &sub : coll)
        core::fill_offsets_for_sub<Index>(sub, offsets, start_i);
    };
    fill_coll(r0);
    fill_coll(r1);
    (fill_coll(rs), ...);

    out.data_buffer().allocate(offsets.back());

    Index start = 0;
    auto copy_coll = [&](const auto &coll) {
      for (const auto &sub : coll)
        core::copy_sub_offset<Index>(sub, out, start);
    };
    copy_coll(r0);
    copy_coll(r1);
    (copy_coll(rs), ...);

    return out;
  }
}

} // namespace tf
