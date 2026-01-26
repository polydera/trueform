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
#include "../core/offset_block_buffer.hpp"
#include "./face_link_like.hpp"
#include "./face_membership_like.hpp"
#include "./structures/compute_face_link.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Stores face adjacency through shared edges.
///
/// For each face, stores the indices of neighboring faces that share an edge.
/// This is used for mesh traversal, connected component detection, and
/// operations that need to walk across face boundaries.
///
/// Requires @ref tf::face_membership to be built first.
///
/// @tparam Index The integer type for face indices.
template <typename Index>
class face_link : public face_link_like<offset_block_buffer<Index, Index>> {
  using base_t = face_link_like<offset_block_buffer<Index, Index>>;

public:
  /// @brief Build face adjacency from face blocks and face membership.
  /// @tparam Range The face blocks range type.
  /// @tparam Policy The face membership policy type.
  /// @param blocks The face blocks (typically from @ref tf::manifold_edge_link).
  /// @param blink The face membership structure.
  template <typename Range, typename Policy>
  auto build(const Range &blocks, const tf::face_membership_like<Policy> &blink)
      -> void {
    base_t::offsets_buffer().allocate(blocks.size() + 1);
    // assume a closed triangular mesh
    base_t::data_buffer().reserve(blocks.size() * 3);
    topology::compute_face_link(blocks, blink, base_t::offsets_buffer(),
                                base_t::data_buffer());
  }
};

} // namespace tf
