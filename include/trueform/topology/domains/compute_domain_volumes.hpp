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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Sum per-fragment Gauss volume contributions into per-domain
/// signed volumes via the fragment-side -> domain map.
///
/// Each fragment `c` contributes `-C` (where `C = contributions[c]`) to
/// the domain bounded on its side 0 (stored normal points INTO this
/// domain) and `+C` to the domain bounded on its side 1 (stored normal
/// points OUT of this domain). The sum over a closed domain's boundary
/// is translation-invariant, so the result is `6 V` for closed interior
/// domains (positive when winding faces outward) and a matching
/// negative value for the unbounded universe domain.
///
/// Sequential by design: `n_components` is small relative to the face
/// count, and the per-component body is just one add + one subtract.
///
/// @param contributions Per-fragment volume contributions
///        (size `n_components`) from
///        @ref tf::topology::domains::compute_volume_contributions_per_fragment.
/// @param domain_of_side Per-fragment-side domain id buffer
///        (size `2 * n_components`) from
///        @ref tf::topology::domains::build_domain_of_side (and any
///        downstream nesting remap).
/// @param n_domains Number of distinct domains.
/// @return `tf::buffer<ContribT>` of size `n_domains` with per-domain
///         signed volume (in units of 6 * volume; sign convention
///         follows the polygons' stored winding).
template <typename ContribT, typename Index>
auto compute_domain_volumes(const tf::buffer<ContribT> &contributions,
                            const tf::buffer<Index> &domain_of_side,
                            Index n_domains) -> tf::buffer<ContribT> {
  tf::buffer<ContribT> domain_volumes;
  domain_volumes.allocate(n_domains);
  tf::parallel_fill(domain_volumes, ContribT(0));

  auto n_components = Index(contributions.size());
  for (Index c = 0; c < n_components; ++c) {
    const auto &C = contributions[c];
    // Side 0 = side stored normal points TO: stored is INWARD of the
    // side-0 domain, so the fragment contributes -C to its volume.
    // Side 1 = side stored normal points AWAY FROM: stored is OUTWARD,
    // so the fragment contributes +C.
    domain_volumes[domain_of_side[2 * c + 0]] -= C;
    domain_volumes[domain_of_side[2 * c + 1]] += C;
  }
  return domain_volumes;
}

} // namespace tf::topology::domains
