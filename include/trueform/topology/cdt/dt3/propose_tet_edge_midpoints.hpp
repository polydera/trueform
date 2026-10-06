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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../../exact/dyadic_blend_scaled.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Selected edges name site slots and admit exact lattice midpoints. Proposals
/// retain selection order and take consecutive fresh canonical names.
template <typename DT, typename Edges, typename Selected, typename Index>
auto propose_tet_edge_midpoints(const DT &dt, const Edges &edges,
                                const Selected &selected, Index first_name,
                                tf::buffer<typename DT::site_type> &proposals)
    -> void {
  using Int = typename DT::int_type;
  const auto sites = dt.sites();
  proposals.allocate(selected.size());
  tf::parallel_for_each(
      tf::make_sequence_range(selected.size()),
      [&](std::size_t i) {
        const auto e = edges[std::size_t(selected[i])];
        const auto &a = sites[std::size_t(e[0])].pt;
        const auto &b = sites[std::size_t(e[1])].pt;
        auto &site = proposals[i];
        site.id = Index(first_name + Index(i));
        for (std::size_t axis = 0; axis < 3; ++axis)
          site.pt[axis] =
              tf::exact::dyadic_blend_scaled<Int>(a[axis], b[axis], 1, 1);
      },
      tf::checked);
}

} // namespace tf::topology::cdt::dt3
