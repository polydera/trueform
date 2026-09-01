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
#include "../../core/edges.hpp"
#include "../../core/points.hpp"
#include "../../core/range.hpp"
#include "../cdt_refine_config.hpp"
#include "../cdt_region_mode.hpp"
#include "./adopt_constrained_delaunay_refinement_faces.hpp"
#include "./adopt_constrained_delaunay_refinement_points.hpp"
#include "./clear_constrained_delaunay_refinement.hpp"
#include "./finalize_constrained_delaunay_refinement_splits.hpp"
#include "./make_constrained_delaunay_refinement_constraints.hpp"
#include "./refine_constrained_delaunay.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner, typename PointsPolicy, typename EdgesPolicy,
          typename BoundaryIterator, std::size_t BoundaryN,
          typename SplittableIterator, std::size_t SplittableN>
auto build_constrained_delaunay_refinement(
    Owner &owner, const tf::points<PointsPolicy> &points,
    const tf::edges<EdgesPolicy> &edges,
    const tf::range<BoundaryIterator, BoundaryN> &is_boundary,
    const tf::range<SplittableIterator, SplittableN> &is_splittable,
    const tf::cdt_refine_config &config, tf::cdt_region_mode mode) -> bool {
  using Index = typename Owner::index_type;

  clear_constrained_delaunay_refinement(owner);
  owner._n_input_points = static_cast<Index>(points.size());
  owner._n_input_edges = static_cast<Index>(edges.size());
  owner._split_encroached = config.split_encroached;

  if (!owner._cdt.build(points, edges, is_boundary,
                        /*split_constraints=*/false, mode))
    return owner._ok = false;

  adopt_constrained_delaunay_refinement_points(owner);
  make_constrained_delaunay_refinement_constraints(owner, edges,
                                                    is_splittable);
  adopt_constrained_delaunay_refinement_faces(owner);
  // Circumcenter refinement is not guaranteed to terminate above 0.45.
  refine_constrained_delaunay(
      owner, std::clamp(double(config.min_quality), 0.0, 0.45));
  finalize_constrained_delaunay_refinement_splits(owner);
  return owner._ok = true;
}

} // namespace tf::topology::cdt
