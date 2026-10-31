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
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./compute_domain_membership.hpp"
#include "./compute_domain_partition.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The partition the domain emitter writes the open fragments
///        through: a component whose two coarse sides are one domain labels
///        its forward side with its own tag, so output `k` is form `k`'s,
///        in stored winding.
template <typename Index>
auto make_open_fragment_partition(const domain_membership<Index> &membership,
                                  const tf::buffer<Index> &tag_of_component,
                                  Index n_tags) -> domain_partition<Index> {
  domain_partition<Index> out;
  out.n_kept = n_tags;
  out.dense_of_domain.allocate(static_cast<std::size_t>(n_tags));
  for (Index t = 0; t < n_tags; ++t)
    out.dense_of_domain[t] = t;

  const auto &domain_of_side = membership.domain_of_side;
  out.side_label.allocate(
      static_cast<std::size_t>(2 * membership.n_components));
  tf::parallel_for_each(
      tf::make_sequence_range(membership.n_components),
      [&](Index c) {
        out.side_label[2 * c + 0] = Index(-1);
        out.side_label[2 * c + 1] =
            domain_of_side[2 * c + 0] == domain_of_side[2 * c + 1]
                ? tag_of_component[c]
                : Index(-1);
      },
      tf::checked);
  return out;
}

} // namespace tf::csg::graph
