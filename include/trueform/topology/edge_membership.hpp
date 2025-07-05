/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/edges.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/segments.hpp"
#include "./edge_orientation.hpp"
#include "./structures/compute_face_membership.hpp"

namespace tf {
template <typename Index>
class edge_membership : public offset_block_buffer<Index, Index> {
  using base_t = offset_block_buffer<Index, Index>;

public:
  template <typename Policy>
  auto build(const tf::edges<Policy> &edges, std::size_t n_unique_ids,
             tf::edge_orientation eo = tf::edge_orientation::bidirectional)
      -> void {
    base_t::offsets_buffer().allocate(n_unique_ids + 1);
    switch (eo) {
    case edge_orientation::forward:
      base_t::data_buffer().allocate(edges.size());
      return build_forward(edges);
    case edge_orientation::reverse:
      base_t::data_buffer().allocate(edges.size());
      return build_reverse(edges);
    case edge_orientation::bidirectional:
      base_t::data_buffer().allocate(edges.size() * 2);
      return build_bidirectional(edges);
    }
  }

  template <typename Policy>
  auto build(const segments<Policy> &segments,
             tf::edge_orientation eo = tf::edge_orientation::bidirectional)
      -> void {
    auto n_unique_ids = segments.points().size();
    build(segments.edges(), n_unique_ids, eo);
  }

private:
  template <typename Range> auto build_forward(const Range &edges) -> void {
    topology::compute_face_membership(
        tf::make_mapped_range(
            edges,
            [](const auto &r) { return std::array<Index, 1>{Index(r[0])}; }),
        base_t::offsets_buffer(), base_t::data_buffer());
  }

  template <typename Range> auto build_reverse(const Range &edges) -> void {
    topology::compute_face_membership(
        tf::make_mapped_range(
            edges,
            [](const auto &r) { return std::array<Index, 1>{Index(r[1])}; }),
        base_t::offsets_buffer(), base_t::data_buffer());
  }

  template <typename Range>
  auto build_bidirectional(const Range &edges) -> void {
    topology::compute_face_membership(edges, base_t::offsets_buffer(),
                                      base_t::data_buffer());
  }
};

} // namespace tf
