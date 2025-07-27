/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/faces.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/polygons.hpp"
#include "./structures/compute_face_membership.hpp"

namespace tf {
template <typename Index>
class face_membership : public offset_block_buffer<Index, Index> {
  using base_t = offset_block_buffer<Index, Index>;

public:
  face_membership() = default;

  template <typename Policy> face_membership(const polygons<Policy> &polygons) {
    build(polygons);
  }

  template <typename Policy>
  auto build(const tf::faces<Policy> &faces, std::size_t n_unique_ids,
             std::size_t total_size) -> void {
    base_t::offsets_buffer().allocate(n_unique_ids + 1);
    base_t::data_buffer().allocate(total_size);
    topology::compute_face_membership(faces, base_t::offsets_buffer(),
                                      base_t::data_buffer());
  }

  template <typename Policy>
  auto build(const polygons<Policy> &polygons) -> void {
    auto n_unique_ids = polygons.points().size();
    constexpr auto n_gons = tf::static_size_v<decltype(polygons[0])>;
    static_assert(n_gons != tf::dynamic_size);
    build(polygons.faces(), n_unique_ids, n_gons * polygons.size());
  }
};

} // namespace tf
