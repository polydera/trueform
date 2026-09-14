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

#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/point_cloud.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Fill the spatial tree of a mesh, if it is missing or stale.
///
/// THE BUILD VERBS' CONTRACT, stated here for the whole family — every other
/// `build_*` header points at this one.
///
/// A build verb is a STATEMENT, not a read: it returns nothing, it is
/// idempotent — a defensive call costs a stamp check — and it builds what the
/// structure stands on before the structure itself. That is what makes
/// "prebuild what you need" sufficient: a caller names the top-level facts a
/// shared region will touch, transitivity does the rest, and no dependency fill
/// is left to race.
///
/// WHAT TO NAME is the READING an entry takes, and a reading names its own
/// structures: `mesh::form()` is the tree, `mesh::topology_form()` is the tree,
/// the face membership and the manifold edge link, and the same two readings on
/// `edge_mesh` and `point_cloud` name theirs. An entry that reads one structure
/// rather than a whole reading says which in its own `@note`. There is no verb
/// for a query FAMILY: the caller names the facts, because the facts are what
/// another thread will find already there.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto build_tree(const mesh<Index, Real, Dims, Ngon> &value) -> void {
  static_cast<void>(value.tree());
}

/// @overload Fill the spatial tree of an edge mesh.
template <typename Index, typename Real, std::size_t Dims>
auto build_tree(const edge_mesh<Index, Real, Dims> &value) -> void {
  static_cast<void>(value.tree());
}

/// @overload Fill the spatial tree of a point cloud.
template <typename Real, std::size_t Dims>
auto build_tree(const point_cloud<Real, Dims> &value) -> void {
  static_cast<void>(value.tree());
}

} // namespace tf::cpp
