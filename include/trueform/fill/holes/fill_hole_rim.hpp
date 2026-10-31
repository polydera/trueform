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
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../topology/boundary_rims.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../hole_fill_config.hpp"
#include "../hole_fill_status.hpp"
#include "../hole_split.hpp"
#include "./fill_chart_hole.hpp"
#include "./fill_table_hole.hpp"
#include "./fill_tiny_hole.hpp"
#include "./hole_lattice.hpp"
#include "./hole_patch.hpp"
#include "./hole_table.hpp"
#include "./make_hole_chart_patch.hpp"
#include "./name_hole_chart_rim.hpp"
#include "./preflight_hole_rim.hpp"
#include "./refine_hole_patch.hpp"
#include "./validate_hole_rim.hpp"
#include <array>
#include <cstddef>

namespace tf::fill {

/// Everything one worker reuses across the rims it takes.
template <typename Index, typename Int, typename RealT>
struct hole_fill_scratch {
  tf::fill::hole_rim_scratch<Index, Int> preflight;
  tf::fill::hole_rim<Index, Int, RealT> rim;
  tf::fill::hole_tiny_scratch<Index, Int> tiny;
  tf::fill::hole_chart_scratch<Index, Int, RealT> chart;
  tf::fill::hole_chart_rim_names<Index, RealT> chart_names;
  tf::fill::hole_table_scratch<Index, Int, RealT> complex;
  tf::fill::hole_table<Index> table;
  tf::fill::hole_patch_scratch<Index> shape;
  tf::fill::hole_patch<Index, RealT> patch;
  tf::fill::hole_refine_scratch<Index> refine;
};

/// Fill one rim, walking the ladder until a tier holds it.
///
/// The tiers are tried in the order of what they can prove: a rim of at most
/// five positions by exhaustive enumeration, then the rim read as a height
/// field over the plane it spans, then the general rim over the
/// tetrahedralization of its sites. A tier that declines leaves nothing
/// behind, so the next one starts on an empty product; `charted` states what
/// the chart made of the rim whichever tier then held it.
///
/// A tier that states a COARSE patch hands it to the refinement driver, which
/// shapes it toward the configured floor and states how it left; the chart
/// samples itself in its own refinement and states that reason instead.
/// `fairs` says which of the two happened, because only a patch this tier
/// refined is one it also fairs.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Int, typename RealT, typename Converter>
auto fill_hole_rim(const tf::polygons<Policy> &polygons,
                   const tf::face_membership_like<MembershipPolicy> &fm,
                   const tf::boundary_rims<Index> &rims, Index rim,
                   const Converter &converter,
                   const tf::hole_fill_config &config, Index n_points,
                   tf::fill::hole_fill_scratch<Index, Int, RealT> &scratch,
                   tf::buffer<std::array<Index, 3>> &triangles,
                   tf::buffer<tf::point<RealT, 3>> &minted,
                   tf::buffer<tf::hole_split<Index>> &splits,
                   tf::hole_chart_status &charted,
                   tf::hole_refine_status &refined, bool &fairs,
                   Index &offending) -> tf::hole_fill_status {
  const auto verdict =
      tf::fill::preflight_hole_rim(polygons, fm, rims, rim, converter,
                                   scratch.preflight, scratch.rim);
  if (verdict.status != tf::hole_fill_status::filled) {
    offending = verdict.edge;
    return verdict.status;
  }

  const auto shaped = [&]() {
    tf::fill::build_hole_patch(scratch.rim, triangles, scratch.shape,
                               scratch.patch);
    const std::size_t coarse = scratch.patch.n_faces();
    refined = tf::fill::refine_hole_patch(
        polygons, fm, n_points, config.min_quality,
        tf::fill::hole_refine_mints_per_triangle * coarse,
        tf::fill::hole_refine_flips_per_triangle * coarse, minted,
        scratch.refine, scratch.patch);
    tf::fill::emit_hole_patch(scratch.patch, scratch.shape, triangles);
    fairs = true;
  };

  if (scratch.rim.size() <= 5 &&
      tf::fill::fill_tiny_hole(polygons, fm, scratch.rim, scratch.tiny,
                               triangles)) {
    shaped();
    return tf::hole_fill_status::filled;
  }

  charted = tf::fill::fill_chart_hole(
      polygons, fm, scratch.rim, n_points,
      tf::fill::hole_lattice_unit(converter), config.split_rim, scratch.chart,
      scratch.chart_names, triangles, minted, splits, refined);
  if (charted == tf::hole_chart_status::admitted ||
      charted == tf::hole_chart_status::rim_plane)
    return tf::hole_fill_status::filled;

  const auto status = tf::fill::fill_table_hole(
      polygons, fm, n_points, config.split_rim, scratch.complex, scratch.table,
      scratch.rim, triangles, minted, splits, offending);
  if (status == tf::hole_fill_status::filled)
    shaped();
  return status;
}

} // namespace tf::fill
