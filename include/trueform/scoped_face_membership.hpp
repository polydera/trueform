/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./implementation/compute_face_membership.hpp"
#include "./offset_block_buffer.hpp"
#include "./polygons.hpp"
#include "./scoped_id.hpp"

namespace tf {
template <typename Index, typename SubIndex = Index>
class scoped_face_membership
    : public offset_block_buffer<Index, tf::scoped_id<Index, SubIndex>> {
  using base_t = offset_block_buffer<Index, tf::scoped_id<Index, SubIndex>>;

public:
  template <typename Range>
  auto build(const Range &blocks, std::size_t n_unique_ids,
             std::size_t total_size) -> void {
    base_t::offsets_buffer().allocate(n_unique_ids + 1);
    base_t::data_buffer().allocate(total_size);
    implementation::compute_face_membership(
        blocks, base_t::offsets_buffer(), base_t::data_buffer(),
        [](auto sub_id, auto block_id) {
          return tf::scoped_id{Index(block_id), SubIndex(sub_id)};
        });
  }

  template <typename Policy>
  auto build(const polygons<Policy> &polygons) -> void {
    auto n_unique_ids = polygons.points().size();
    constexpr auto n_gons = tf::static_size_v<decltype(polygons[0])>;
    static_assert(n_gons != tf::dynamic_size);
    build(polygons, n_unique_ids, n_gons * polygons.size());
  }
};

} // namespace tf
