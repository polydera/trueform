/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/faces.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/views/drop.hpp"
#include "../core/views/slide_range.hpp"
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

template <typename Index>
class manifold_edge_link<Index, tf::dynamic_size>
    : public manifold_edge_link_like<
          offset_block_buffer<Index, manifold_edge_peer<Index>>> {
  using base_t = manifold_edge_link_like<
      offset_block_buffer<Index, manifold_edge_peer<Index>>>;

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
    if (!blocks.size())
      return;
    base_t::offsets_buffer().allocate(blocks.size() + 1);
    base_t::offsets_buffer()[0] = 0;
    tf::parallel_transform(
        blocks, tf::drop(base_t::offsets_buffer(), 1),
        [](const auto &block) { return block.size(); }, tf::checked);
    for (auto &&[a, b] : tf::make_slide_range<2>(base_t::offsets_buffer()))
      b += a;
    base_t::data_buffer().allocate(base_t::offsets_buffer().back());
    topology::compute_manifold_edge_link(blocks, blink, *this);
  }
};

} // namespace tf
