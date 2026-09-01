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
#include "../../core/point.hpp"
#include "../../exact/incircle.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "./build_divide_and_conquer.hpp"
#include "./build_divide_and_conquer_parallel.hpp"
#include "./delaunay_incircle_sign.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt {

template <bool NarrowOrientation, bool NarrowIncircle, typename ExecutionPolicy,
          typename Owner>
auto build_prepared_delaunay_topology(Owner &owner,
                                      typename Owner::index_type none)
    -> tf::topology::cdt::delaunay_build_result<typename Owner::index_type> {
  using Index = typename Owner::index_type;
  const auto point = [&](Index index) {
    const auto &site = owner._sites[std::size_t(index)];
    return tf::make_point(site.x, site.y);
  };
  const auto orientation = [&](Index a, Index b, Index c) {
    if constexpr (NarrowOrientation) {
      using T1 = typename tf::exact::meta<typename Owner::int_type>::T1;
      const auto &pa = owner._sites[std::size_t(a)];
      const auto &pb = owner._sites[std::size_t(b)];
      const auto &pc = owner._sites[std::size_t(c)];
      const T1 determinant = (T1(pb.x) - pa.x) * (T1(pc.y) - pa.y) -
                             (T1(pb.y) - pa.y) * (T1(pc.x) - pa.x);
      return determinant > 0 ? 1 : determinant < 0 ? -1 : 0;
    } else {
      return tf::exact::orient2d_sign(point(a), point(b), point(c));
    }
  };
  const auto circle = [&](Index a, Index b, Index c, Index d) {
    if constexpr (NarrowIncircle)
      return tf::topology::cdt::delaunay_incircle_sign_t1(
                 point(a), point(b), point(c), point(d)) > 0;
    else
      return tf::exact::incircle_sign(point(a), point(b), point(c), point(d)) >
             0;
  };
  tf::topology::cdt::delaunay_build_result<Index> result;
  if constexpr (ExecutionPolicy::parallel) {
    result = tf::topology::cdt::build_divide_and_conquer_parallel<Index>(
        owner._sites, owner._edges, owner._keys, none, orientation, circle,
        Owner::k_darts_per_site_reserve, owner._partition_nodes,
        owner._partition_leaves);
  } else {
    result = tf::topology::cdt::build_divide_and_conquer<Index, true>(
        owner._sites, owner._edges, owner._keys, none, orientation, circle);
  }
  Owner::retention_policy::finish_delaunay_edges(owner._edges, none);
  return result;
}

} // namespace tf::topology::cdt
