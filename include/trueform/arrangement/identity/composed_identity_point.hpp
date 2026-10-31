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

#include "../../exact/vertex.hpp"
#include "../../intersect/graph/final_identity_point.hpp"
#include "./identity_substrate.hpp"

#include <cstdint>

namespace tf::arrangement {

/// Where `(tag, id)` stands at the composed layer `P∘L` (@ref
/// identity_substrate::composed_target): a created id below the plane value's
/// frozen `n_base_created()` is stated by the world, the rest by `arrangement`,
/// and one enclosing chain spans both.
template <typename Index, typename Int, typename World, typename Arrangement,
          typename GetMeshPoint>
auto composed_identity_point(
    const identity_substrate<Index, Int> &identities, const World &world,
    const Arrangement &arrangement, const GetMeshPoint &get_mesh_point,
    std::int16_t tag, Index id,
    const tf::intersect::graph::evaluated_identity<Index> *mints = nullptr)
    -> tf::exact::pt3<Int> {
  using pt3_t = tf::exact::pt3<Int>;
  return tf::intersect::graph::final_identity_point<pt3_t>(
      identities.local_merges().size() != 0 ||
          identities.plane_merges().size() != 0,
      [&identities](Index source, Index key) {
        return identities.composed_target(source, key);
      },
      world.vertex_offsets(), tag, id, mints,
      [&](std::int16_t at_tag, Index at,
          const tf::intersect::graph::evaluated_identity<Index> *self)
          -> pt3_t {
        if (at_tag >= 0)
          return pt3_t(get_mesh_point(int(at_tag), at));
        const auto read = [&](std::int16_t read_tag, Index point) {
          return composed_identity_point(identities, world, arrangement,
                                         get_mesh_point, read_tag, point,
                                         self);
        };
        if (at >= arrangement.n_base_created())
          return arrangement.statement_point(identities, at, read);
        return world.statement_point(
            identities, at, read, [&](std::int16_t, Index point) {
              return world.intersection_point(point, get_mesh_point);
            });
      });
}

} // namespace tf::arrangement
