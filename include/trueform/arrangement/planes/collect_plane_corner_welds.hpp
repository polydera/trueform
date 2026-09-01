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

#include "../../core/range.hpp"
#include "./collect_plane_flat_welds.hpp"

namespace tf::arrangement {

/// SURGERY. Name one plane's corners off the triangulation that produced
/// them: every prepared identity the projection could not tell apart from
/// another states its weld evidence here.
template <typename Index, typename Local, typename Mesh>
auto collect_plane_corner_welds(Local &local, const Mesh &mesh, Index plane,
                                Index n_flat) -> bool {
  const auto n_final = Index(mesh.points().size());
  return collect_plane_flat_welds(
      tf::make_range(local.ends), mesh.index_map().f(),
      mesh.index_map().kept_ids(), Index(mesh.index_map().f().size()), n_final,
      n_flat, plane, local.rep, local.welds);
}

} // namespace tf::arrangement
