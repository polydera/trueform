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
#include "../core/tetras.hpp"
#include "./manifold_facet_link_like.hpp"
#include "./manifold_facet_peer.hpp"
#include "./structures/compute_manifold_facet_link.hpp"
#include "./tet_membership_like.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Per tet and per facet slot `k`, the @ref tf::manifold_facet_peer
/// across the facet opposite corner `k`.
///
/// The manifold edge link one dimension up, found through the tet membership.
///
/// @tparam Index The integer type for tet indices.
template <typename Index>
class manifold_facet_link : public manifold_facet_link_like<
                                blocked_buffer<manifold_facet_peer<Index>, 4>> {
  using base_t =
      manifold_facet_link_like<blocked_buffer<manifold_facet_peer<Index>, 4>>;

public:
  manifold_facet_link() = default;

  /// @brief Construct and build from the tets' corners and their membership.
  template <typename Policy, typename Policy1>
  manifold_facet_link(const tf::tetras<Policy> &tetras,
                      const tf::tet_membership_like<Policy1> &blink) {
    build(tetras, blink);
  }

  /// @brief Build from the tets' corner blocks and their membership.
  template <typename Range, typename Policy1>
  auto build(const Range &blocks, const tf::tet_membership_like<Policy1> &blink)
      -> void {
    base_t::data_buffer().allocate(blocks.size() * 4);
    topology::compute_manifold_facet_link(blocks, blink, *this);
  }
};

} // namespace tf
