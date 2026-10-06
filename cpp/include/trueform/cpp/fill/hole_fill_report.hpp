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

#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/fill/hole_fill_result.hpp"

#include <cstdint>
#include <memory>
#include <type_traits>

namespace tf::cpp {
/// @brief What `fill_holes` made of a mesh's rims, in the facade's carriers.
///
/// `result` is the fill itself and the one authority `filled_mesh` replays:
/// every lane below is a view of its storage, so the stitch reads the patches,
/// the split table and the face plans the fill stated and nothing restated
/// beside them. Where a record is not a scalar the lane is a copy: the
/// statuses, narrowed to `std::int8_t` from the enums they name, and
/// `splits`, one row per record.
///
/// The carrier is the group: row `g` of a per-group lane and block `g` of a
/// jagged lane belong to group `g`, and a jagged lane is its offsets beside
/// its rows, block `g` being rows `[offsets[g], offsets[g + 1])`. A corner is
/// a flat id: below `number_of_points` a vertex of the mesh, at or above it a
/// row of `minted_points`, whose blocks follow in group order.
template <typename Index, typename Real> struct hole_fill_report {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "hole_fill_report requires an unqualified supported index "
                "type");

  std::shared_ptr<const tf::hole_fill_result<Index, Real>> result;
  /// @brief Shaped [G]: per group its @ref tf::hole_fill_status.
  nd_array<std::int8_t> status;
  /// @brief Shaped [G]: per group its @ref tf::hole_refine_status.
  nd_array<std::int8_t> refined;
  /// @brief Shaped [G]: per group its @ref tf::hole_fair_status.
  nd_array<std::int8_t> faired;
  /// @brief Shaped [G]: per group the largest angle across its seam, `-1`
  /// where it was not computed.
  nd_array<Real> seam_max_angle;
  /// @brief Shaped [G, 2]: per group the `(rim, edge)` its status speaks of,
  /// `(-1, -1)` where there is none. The rim is a `boundary_rims` id.
  nd_array<Index> offending;
  /// @brief Shaped [G + 1]: per group its run of the rims the call filled.
  nd_array<Index> group_offsets;
  nd_array<Index> triangle_offsets;
  /// @brief Shaped [T, 3]: the patches, in flat ids.
  nd_array<Index> triangles;
  nd_array<Index> minted_point_offsets;
  /// @brief Shaped [M, 3]: the points the patches minted.
  nd_array<Real> minted_points;
  /// @brief Shaped [S, 5]: every original edge the fill cut, as
  /// `(face, v0, v1, point, parameter)` with `v0 < v1` and the parameter the
  /// numerator over @ref tf::hole_split_scale measured from `v0`.
  nd_array<Index> splits;
  /// @brief Shaped [P]: the carrying faces those splits reached, ascending.
  nd_array<Index> plan_faces;
  nd_array<Index> plan_triangle_offsets;
  /// @brief Shaped [Q, 3]: per plan face the triangles it becomes.
  nd_array<Index> plan_triangles;
  Index number_of_points = 0;
  Index number_of_faces = 0;
};

} // namespace tf::cpp
