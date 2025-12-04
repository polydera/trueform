/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/reduce.hpp"
#include "../core/faces.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/polygons.hpp"
#include "../core/views/mapped_range.hpp"
#include "./face_membership_like.hpp"
#include "./structures/compute_face_membership.hpp"

namespace tf {
template <typename Index>
class face_membership
    : public face_membership_like<offset_block_buffer<Index, Index>> {
  using base_t = face_membership_like<offset_block_buffer<Index, Index>>;

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
    if constexpr (n_gons != tf::dynamic_size) {
      build(polygons.faces(), n_unique_ids, n_gons * polygons.size());
    } else {
      auto sizes = tf::make_mapped_range(
          polygons.faces(), [](const auto &face) { return face.size(); });
      auto total_size = tf::reduce(
          sizes, [](auto a, auto b) { return a + b; }, std::size_t{0},
          tf::checked);
      build(polygons.faces(), n_unique_ids, total_size);
    }
  }
};

} // namespace tf
