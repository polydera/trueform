/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_apply.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/edges.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/polygons.hpp"
#include "../core/segments.hpp"
#include "../core/views/enumerate.hpp"
#include "./edge_membership.hpp"
#include "./face_membership.hpp"
#include "./scoped_face_membership.hpp"
#include "./structures/compute_vertex_link.hpp"

namespace tf {
template <typename Index>
class vertex_link : public offset_block_buffer<Index, Index> {
  using base_t = offset_block_buffer<Index, Index>;

public:
  template <typename Policy>
  auto build(const tf::polygons<Policy> &polygons,
             const tf::face_membership<Index> &blink) -> void {
    base_t::offsets_buffer().allocate(blink.size() + 1);
    // perfect mesh has valence 6
    base_t::data_buffer().reserve(blink.size() * 4);
    topology::compute_vertex_link(polygons.faces(), blink,
                                  base_t::offsets_buffer(),
                                  base_t::data_buffer());
  }

  template <typename Policy, typename SubIndex>
  auto build(const tf::polygons<Policy> &polygons,
             const tf::scoped_face_membership<Index, SubIndex> &blink) -> void {
    base_t::offsets_buffer().allocate(blink.size() + 1);
    // perfect mesh has valence 6
    base_t::data_buffer().reserve(blink.size() * 4);
    topology::compute_vertex_link(polygons.faces(), blink,
                                  base_t::offsets_buffer(),
                                  base_t::data_buffer());
  }

  template <typename Policy>
  auto build(const tf::edges<Policy> &edges, std::size_t n_unique_ids,
             tf::edge_orientation eo = tf::edge_orientation::bidirectional) {
    auto &as_em =
        static_cast<edge_membership<Index> &>(static_cast<base_t &>(*this));
    as_em.build(edges, n_unique_ids, eo);
    switch (eo) {
    case tf::edge_orientation::forward:
      return tf::parallel_copy(
          tf::make_indirect_range(
              as_em.data_buffer(),
              tf::make_mapped_range(edges, [](const auto &r) { return r[1]; })),
          as_em.data_buffer());
    case tf::edge_orientation::reverse:
      return tf::parallel_copy(
          tf::make_indirect_range(
              as_em.data_buffer(),
              tf::make_mapped_range(edges, [](const auto &r) { return r[0]; })),
          as_em.data_buffer());
    case tf::edge_orientation::bidirectional:
      tf::parallel_apply(
          tf::enumerate(as_em),
          [&](auto pair) {
            auto &&[id, block] = pair;
            for (auto &edge_id : block) {
              const auto &edge = edges[edge_id];
              edge_id = edge[edge[0] == Index(id)];
            }
          },
          tf::checked);
    }
  }

  template <typename Policy>
  auto build(const tf::segments<Policy> &segments,
             tf::edge_orientation eo = tf::edge_orientation::bidirectional) {
    build(segments.edges(), segments.points().size(), eo);
  }
};

} // namespace tf
