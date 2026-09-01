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
#include "./build_prepared_delaunay_topology.hpp"
#include "./delaunay_execution_policy.hpp"

namespace tf::topology::cdt {

/// Build the same exact Delaunay topology for every retention policy. The
/// policy only selects the edge carrier and whether its cold fields exist.
template <typename ExecutionPolicy = serial_delaunay_execution_policy,
          typename Owner>
auto build_delaunay_topology(Owner &owner, typename Owner::index_type none)
    -> tf::topology::cdt::delaunay_build_result<typename Owner::index_type> {
  if (owner._orientation_fits_t1)
    return owner._incircle_inner_fits_t1
               ? build_prepared_delaunay_topology<true, true, ExecutionPolicy>(
                     owner, none)
               : build_prepared_delaunay_topology<true, false, ExecutionPolicy>(
                     owner, none);
  return owner._incircle_inner_fits_t1
             ? build_prepared_delaunay_topology<false, true, ExecutionPolicy>(
                   owner, none)
             : build_prepared_delaunay_topology<false, false, ExecutionPolicy>(
                   owner, none);
}

} // namespace tf::topology::cdt
