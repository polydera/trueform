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
#include "../../core/coordinate_type.hpp"
#include "../../core/polygons.hpp"
#include "../../core/resolved_output_real.hpp"
#include "../../core/return_report.hpp"
#include "../../core/static_size.hpp"
#include "../entrance/make_tet_entrance.hpp"
#include "../entrance/measure_tet_corner_need.hpp"
#include "../entrance/tet_entrance_rung_types.hpp"
#include "../tet_mesh_config.hpp"
#include "../tet_mesh_report.hpp"
#include "./mesh_tet_entrance.hpp"
#include <type_traits>

namespace tf::tetmesh {

/// Mesh the solid `polygons` winds on the cheapest rung whose lattice holds
/// its precision, restricted to `Int` when stated: its entrance
/// (@ref make_tet_entrance), its planes pooled under `config.tolerance`, then
/// the mesher under `config` (@ref mesh_tet_entrance), emitted in
/// `OutputCoordinateType` with what `ReportTag` and `ReturnTag` ask.
template <typename Int, typename OutputCoordinateType, typename Policy,
          typename ReportTag, typename ReturnTag>
auto mesh_tet_surface(const tf::polygons<Policy> &polygons,
                      const tf::tet_mesh_config &config, ReportTag report_tag,
                      ReturnTag tag) {
  using RealT = tf::coordinate_type<Policy>;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, RealT>;
  static_assert(std::is_floating_point_v<RealOut>,
                "a tet mesh is emitted in floating-point coordinates");
  constexpr bool home = tet_entrance_rung_admits<0, Int, RealT>;
  static_assert(home || tet_entrance_rung_admits<1, Int, RealT>,
                "no rung's lattice holds the input's precision");
  static_assert(tf::static_size_v<decltype(polygons.faces()[0])> == 3,
                "a tet mesh is made of a triangle mesh");
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using types = tet_entrance_rung_types<home ? 0 : 1>;
  tf::tet_mesh_report report;
  if constexpr (std::is_same_v<ReportTag, tf::return_report_t>) {
    const auto need = measure_tet_corner_need(polygons);
    report.corner_bits_needed = need.corner_bits;
    report.min_angle = need.min_angle;
  }
  report.rung = home ? 0 : 1;
  return mesh_tet_entrance<RealOut>(
      polygons,
      make_tet_entrance<Index, typename types::int_type,
                        typename types::coord_type>(polygons, config.tolerance),
      report, config, report_tag, tag);
}

} // namespace tf::tetmesh
