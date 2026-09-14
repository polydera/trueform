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
#include "../core/buffer.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/range.hpp"
#include "./tree/winding_block.hpp"
#include "./tree_like.hpp"
#include "./winding/aggregate_winding_leaves.hpp"
#include "./winding/finalize_winding_moments.hpp"
#include "./winding/raw_winding_moment.hpp"
#include "./winding/sweep_winding_moments.hpp"
#include <cstdint>

namespace tf {

/// @ingroup spatial_structures
/// @brief The winding moments of a tree: per node the expansion to
/// Taylor order two — the dipole, the mixed second moment
/// ∫ n_i (p - p̃)_j dA, and the quadrupole third-moment tier
/// ∫ n_i (p - p̃)_j (p - p̃)_k dA — aligned to the tree's own node
/// array (Barill, Dickson, Schmidt, Levin & Jacobson, "Fast Winding
/// Numbers for Soups and Clouds", SIGGRAPH 2018). The tiers ship
/// together: the middle term alone measures worse than the dipole on
/// curved clusters, and this sentence is why "order 2" is enumerated
/// here rather than assumed.
///
/// Built over a built tree and the polygons it was built over; a moment
/// row is meaningful only beside the node row it mirrors, so a tree
/// rebuild invalidates the moments with it. Attach to a form beside the
/// tree — `polygons | tf::tag(tree) | tf::tag(moments)` — and query
/// through @ref tf::winding_number.
template <typename RealT> class winding_moments {
public:
  template <typename TreePolicy, typename Polygons>
  auto build(const tf::tree_like<TreePolicy> &tree, const Polygons &polygons)
      -> void {
    static_assert(tf::coordinate_dims_v<Polygons> == 3,
                  "winding moments require 3D polygons");
    const auto nodes = tree.nodes();
    tf::buffer<spatial::raw_winding_moment> raw;
    raw.allocate(nodes.size());
    spatial::aggregate_winding_leaves(raw.begin(), nodes, tree.ids(), polygons);
    spatial::sweep_winding_moments(raw.begin(), nodes);
    spatial::finalize_winding_moments(_blocks, _tickets, raw.begin(), nodes);
  }

  auto moments() const -> spatial::winding_moments_view<RealT> {
    return {tf::make_range(_blocks.begin(), _blocks.end()),
            tf::make_range(_tickets.begin(), _tickets.end())};
  }

private:
  tf::buffer<spatial::winding_block<RealT>> _blocks;
  tf::buffer<std::int32_t> _tickets;
};

/// @ingroup spatial_structures
/// @brief Build the winding moments of a built tree over the polygons
/// it was built over.
template <typename TreePolicy, typename Polygons>
auto make_winding_moments(const tf::tree_like<TreePolicy> &tree,
                          const Polygons &polygons) {
  using real_t = tf::coordinate_type<typename TreePolicy::bv_type>;
  winding_moments<real_t> out;
  out.build(tree, polygons);
  return out;
}

} // namespace tf
