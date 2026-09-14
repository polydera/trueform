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
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/resolved_output_real.hpp"
#include "./impl/emit_in_stated_frame.hpp"
#include "./impl/flying_dc.hpp"
#include "./impl/flying_edges.hpp"
#include "./isosurface_config.hpp"
#include "./volume.hpp"
#include <type_traits>

namespace tf {

/// @ingroup volume
/// @brief Extract the isosurface of a scalar volume as a triangle mesh.
///
/// Produces the level set `{ p : field(p) == iso }` as a welded, indexed
/// triangle mesh in the frame the volume states — its local frame, or the
/// tagged one it carries. For a signed distance field the
/// natural choice is `iso == 0`, recovering the zero-level surface.
///
/// This is the stable public entry point for isosurfacing, backed by a
/// Flying Edges extractor — row-parallel, producing the welded indexed
/// mesh directly with no triangle soup and no post-hoc vertex weld.
///
/// A corner is treated as inside when `sample < iso`. With the SDF sign
/// convention (negative inside) this yields outward-facing triangle windings.
///
/// The call decides and emits in ONE type: `OutputCoordinateType`, or the
/// volume's coordinate type when the caller states none — a floating type,
/// refused otherwise. A sample of any storage type is read through one cast
/// into it, so the classifier and the crossing that follows stand on the
/// same number.
///
/// @tparam Index The index type for the output mesh (default: int).
/// @tparam OutputCoordinateType The coordinate type of the emitted mesh
///         (default: the volume's coordinate type).
/// @param vol The scalar volume.
/// @param iso The isovalue to extract (default 0 — the SDF zero level set). A
///        nonzero value extracts an offset surface: for an SDF, `+d` inflates
///        by `d` and `-d` deflates by `d`.
/// @return A `polygons_buffer` triangle mesh; empty if the grid is too small or
///         the isovalue is not crossed.
template <typename Index = int, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_isosurface(const tf::volume<Policy> &vol,
                tf::resolved_output_real_t<OutputCoordinateType,
                                           tf::coordinate_type<Policy>>
                    iso = 0) {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "an isosurface is the level set of a 3D volume; a 2D one is "
                "asked for its isocontours");
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>;
  static_assert(std::is_floating_point_v<RealOut>,
                "the extraction decides in a floating type: state "
                "OutputCoordinateType, or give the grid a floating "
                "coordinate type");
  auto mesh = tf::volume_detail::flying_edges<Index, RealOut>(vol, iso);
  tf::volume_detail::emit_mesh_in_stated_frame(mesh, vol);
  return mesh;
}

/// @ingroup volume
/// @brief Extract the isosurface of a scalar volume with an explicit method.
///
/// Same surface contract as the two-argument entry — a welded, indexed
/// triangle mesh in the frame the volume states, a corner inside when
/// `sample < iso`, and
/// outward winding for an SDF. @ref tf::isosurface_method::dual_contouring
/// additionally fits each vertex to the field's crossings, so creases and
/// corners survive; its output is manifold by construction, including where two
/// contour arcs of one grid face join the same pair of cells.
///
/// @param vol The scalar volume.
/// @param iso The isovalue to extract.
/// @param config Method and its parameters; implicitly constructible from an
///        @ref tf::isosurface_method.
template <typename Index = int, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_isosurface(const tf::volume<Policy> &vol,
                tf::resolved_output_real_t<OutputCoordinateType,
                                           tf::coordinate_type<Policy>>
                    iso,
                const tf::isosurface_config &config) {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "an isosurface is the level set of a 3D volume; a 2D one is "
                "asked for its isocontours");
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>;
  static_assert(std::is_floating_point_v<RealOut>,
                "the extraction decides in a floating type: state "
                "OutputCoordinateType, or give the grid a floating "
                "coordinate type");
  if (config.method == tf::isosurface_method::dual_contouring) {
    tf::volume_detail::dual_contouring<Index, RealOut> dc;
    auto mesh = dc.build(vol, iso, config);
    tf::volume_detail::emit_mesh_in_stated_frame(mesh, vol);
    return mesh;
  }
  auto mesh = tf::volume_detail::flying_edges<Index, RealOut>(vol, iso);
  tf::volume_detail::emit_mesh_in_stated_frame(mesh, vol);
  return mesh;
}

} // namespace tf
