/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/blocked_buffer.hpp"
#include "../core/faces.hpp"
#include "./manifold_edge_link_like.hpp"
#include "./structures/compute_manifold_edge_link.hpp"

namespace tf {
template <typename Index, std::size_t N>
class manifold_edge_link : public manifold_edge_link_like<
                               blocked_buffer<manifold_edge_peer<Index>, N>> {
  using base_t =
      manifold_edge_link_like<blocked_buffer<manifold_edge_peer<Index>, N>>;

public:
  manifold_edge_link() = default;

  template <typename Policy, typename Policy1>
  manifold_edge_link(const tf::faces<Policy> &faces,
                     const tf::face_membership_like<Policy1> &blink) {
    build(faces, blink);
  }

  template <typename Range, typename Policy1>
  auto build(const Range &blocks,
             const tf::face_membership_like<Policy1> &blink) -> void {
    base_t::data_buffer().allocate(blocks.size() * N);
    topology::compute_manifold_edge_link(blocks, blink, *this);
  }
};

} // namespace tf
