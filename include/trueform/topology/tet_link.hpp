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
#include "./structures/compute_tet_link.hpp"
#include "./tet_link_like.hpp"
#include "./tet_membership_like.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Per tet, every other tet sharing one of its facets, facet slot by
/// facet slot.
///
/// The face link one dimension up, found through the tet membership.
///
/// @tparam Index The integer type for tet indices.
template <typename Index>
class tet_link : public tet_link_like<offset_block_buffer<Index, Index>> {
  using base_t = tet_link_like<offset_block_buffer<Index, Index>>;

public:
  /// @brief Build from the tets' corner blocks and their membership.
  template <typename Range, typename Policy>
  auto build(const Range &blocks, const tf::tet_membership_like<Policy> &blink)
      -> void {
    base_t::offsets_buffer().allocate(blocks.size() + 1);
    base_t::data_buffer().reserve(blocks.size() * 4);
    topology::compute_tet_link(blocks, blink, base_t::offsets_buffer(),
                               base_t::data_buffer());
  }
};

} // namespace tf
