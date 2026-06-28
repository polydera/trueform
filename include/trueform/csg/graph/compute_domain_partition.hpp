/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../cut/arrangements/arrangement_descriptor.hpp"
#include <cstdint>

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Per component-side dense kept-domain label, driving per-domain
///        emission (one output mesh per kept domain).
///
/// Unlike @ref tf::csg::graph::compute_chosen_sides, which collapses to a
/// single rev/fwd solid, this keys on the domain id bounding each side:
/// side 0 = reverse winding, side 1 = forward. Self-merged components
/// (`d0 == d1`, the open fragments the arrangement fused) are non-separating
/// internal flaps and label both sides `-1` so emission skips them.
template <typename Index> struct domain_partition {
  tf::buffer<Index> dense_of_domain; // size n_domains; kept -> [0,n_kept), else -1
  tf::buffer<Index> side_label;      // size 2*n_components; dense kept id, or -1
  Index n_kept = 0;
};

/// @brief Dense-remap kept domains to `[0, n_kept)`, then label each
///        component-side with the dense id of the domain it bounds (or -1).
///
/// @param desc Arrangement descriptor (`domain_of_side` + `n_components`
///             derived from `bundle_of_component`).
/// @param keep Per-domain keep flag (`keep.size() == n_domains`).
template <typename Index>
auto compute_domain_partition(const tf::cut::arrangement_descriptor<Index> &desc,
                              const tf::buffer<bool> &keep)
    -> domain_partition<Index> {
  domain_partition<Index> out;
  const Index n_domains = static_cast<Index>(keep.size());
  out.dense_of_domain.allocate(static_cast<std::size_t>(n_domains));
  Index next = 0;
  for (Index d = 0; d < n_domains; ++d)
    out.dense_of_domain[d] = keep[d] ? next++ : Index(-1);
  out.n_kept = next;

  const Index n_components =
      static_cast<Index>(desc.bundle_of_component.size());
  out.side_label.allocate(static_cast<std::size_t>(2 * n_components));
  tf::parallel_for_each(tf::make_sequence_range(n_components), [&](Index c) {
    const Index d0 = desc.domain_of_side[2 * c + 0];
    const Index d1 = desc.domain_of_side[2 * c + 1];
    if (d0 == d1) { // self-merged open fragment: non-separating, skip both sides
      out.side_label[2 * c + 0] = Index(-1);
      out.side_label[2 * c + 1] = Index(-1);
      return;
    }
    for (Index s = 0; s < 2; ++s) {
      const Index d = desc.domain_of_side[2 * c + s];
      out.side_label[2 * c + s] =
          (d >= 0 && d < n_domains) ? out.dense_of_domain[d] : Index(-1);
    }
  });
  return out;
}

/// @brief Overload taking a coarsened `domain_of_side` directly.
///
/// Same labelling as the descriptor overload, but driven by an explicit
/// per-side domain id buffer (e.g. the open-merge-coarsened sides from
/// @ref compute_domain_membership) rather than `desc.domain_of_side`.
/// The `d0 == d1` self-merged-fragment skip still applies on the
/// coarsened sides. `keep.size() == n_coarse`.
template <typename Index>
auto compute_domain_partition(const tf::buffer<Index> &domain_of_side,
                              Index n_components, const tf::buffer<bool> &keep)
    -> domain_partition<Index> {
  domain_partition<Index> out;
  const Index n_domains = static_cast<Index>(keep.size());
  out.dense_of_domain.allocate(static_cast<std::size_t>(n_domains));
  Index next = 0;
  for (Index d = 0; d < n_domains; ++d)
    out.dense_of_domain[d] = keep[d] ? next++ : Index(-1);
  out.n_kept = next;

  out.side_label.allocate(static_cast<std::size_t>(2 * n_components));
  tf::parallel_for_each(tf::make_sequence_range(n_components), [&](Index c) {
    const Index d0 = domain_of_side[2 * c + 0];
    const Index d1 = domain_of_side[2 * c + 1];
    if (d0 == d1) {
      out.side_label[2 * c + 0] = Index(-1);
      out.side_label[2 * c + 1] = Index(-1);
      return;
    }
    for (Index s = 0; s < 2; ++s) {
      const Index d = domain_of_side[2 * c + s];
      out.side_label[2 * c + s] =
          (d >= 0 && d < n_domains) ? out.dense_of_domain[d] : Index(-1);
    }
  });
  return out;
}

} // namespace tf::csg::graph
