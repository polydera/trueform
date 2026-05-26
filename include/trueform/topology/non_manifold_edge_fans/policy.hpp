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
#include "../../core/range.hpp"
#include <utility>

namespace tf::topology {

/// @ingroup topology_components
/// @brief Storage-agnostic policy for non-manifold edge fans.
///
/// At each non-manifold edge, three or more faces meet. This policy
/// carries that information as two parallel containers:
///   - `edges`: per-edge endpoint pair (vertex ids).
///   - `faces`: per-edge offset-block of incident face ids
///     (representative face id first, then its neighbours across the
///     edge).
///
/// Both containers may be owning (e.g. `tf::blocked_buffer<Index, 2>` and
/// `tf::offset_block_buffer<Index, Index>`) or views over external
/// storage, so the policy is parameterized over either.
///
/// @tparam EdgesRange The container/view of endpoint pairs.
/// @tparam FacesRange The container/view of per-edge face blocks.
template <typename EdgesRange, typename FacesRange>
struct non_manifold_edge_fans_policy {
  using edges_range_type = EdgesRange;
  using faces_range_type = FacesRange;

  EdgesRange edges;
  FacesRange faces;

  non_manifold_edge_fans_policy() = default;

  non_manifold_edge_fans_policy(EdgesRange e, FacesRange f)
      : edges{std::move(e)}, faces{std::move(f)} {}
};

/// @ingroup topology_components
/// @brief Wrap external edges + faces ranges as a view policy.
template <typename EdgesIn, typename FacesIn>
auto make_non_manifold_edge_fans_policy(EdgesIn &&e, FacesIn &&f) {
  auto edges = tf::make_range(static_cast<EdgesIn &&>(e));
  auto faces = tf::make_range(static_cast<FacesIn &&>(f));
  using policy_t = non_manifold_edge_fans_policy<std::decay_t<decltype(edges)>,
                                                 std::decay_t<decltype(faces)>>;
  return policy_t{edges, faces};
}

} // namespace tf::topology
