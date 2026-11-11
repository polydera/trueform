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
#include "../topology/cdt/dt3/tet_execution_tuning.hpp"
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup tetmesh
/// @brief The facts of one @ref tf::make_tet_mesh call, the 2D CDT's shape
///        (@ref tf::cdt_config).
///
/// `split_constraints` resolves the surface: a crossing the recovery meets
/// is cut in a drain. `false` takes the surface as given, and a crossing in
/// it refuses. `tolerance`, in the input's coordinates, is plane pooling
/// and nothing else: the entrance pools faces that one plane within it
/// could hold onto that plane, and builds no tree and no link for it
/// (@ref tf::exact::input_lattice::build_pooled). `site_budget_ratio`
/// bounds the sites a run may hold per site its entrance and its drains
/// state, a drain's points never refused by it: a split or face point of
/// the recovery that would pass the bound refuses as `budget`
/// (@ref tf::tet_mesh_report::refused_budget). It is set by name.
/// Implicitly constructible from `bool` alone; a number converts to
/// nothing.
struct tet_mesh_config {
  bool split_constraints = true;
  double tolerance = 0.0;
  std::size_t site_budget_ratio =
      tf::topology::cdt::dt3::tet_execution_tuning::site_budget_ratio;

  constexpr tet_mesh_config() = default;
  template <typename Split,
            std::enable_if_t<std::is_same_v<Split, bool>, int> = 0>
  constexpr tet_mesh_config(Split split) : split_constraints(split) {}
  template <typename Split,
            std::enable_if_t<std::is_same_v<Split, bool>, int> = 0>
  constexpr tet_mesh_config(Split split, double tolerance)
      : split_constraints(split), tolerance(tolerance) {}
};

} // namespace tf
