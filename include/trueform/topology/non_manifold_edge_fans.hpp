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
#include "../core/blocked_buffer.hpp"
#include "../core/offset_block_buffer.hpp"
#include "./non_manifold_edge_fans_like.hpp"

namespace tf {

/// @ingroup topology_components
/// @brief Owning container for non-manifold edge fans.
///
/// Carries per-edge endpoint pairs in `edges` and per-edge incident
/// face blocks in `faces` (representative face first, then neighbours
/// across the edge). Iterate `edges` and `faces` together — same row
/// order, both indexed by non-manifold-edge id.
///
/// Build via @ref tf::make_non_manifold_edge_fans.
///
/// @tparam Index The integer type for vertex and face indices.
template <typename Index>
class non_manifold_edge_fans
    : public non_manifold_edge_fans_like<
          tf::topology::non_manifold_edge_fans_policy<
              tf::blocked_buffer<Index, 2>,
              tf::offset_block_buffer<Index, Index>>> {
  using base_t = non_manifold_edge_fans_like<
      tf::topology::non_manifold_edge_fans_policy<
          tf::blocked_buffer<Index, 2>,
          tf::offset_block_buffer<Index, Index>>>;

public:
  using base_t::base_t;
};

} // namespace tf
