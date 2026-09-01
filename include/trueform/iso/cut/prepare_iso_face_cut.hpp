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

#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/signed_area.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/loop.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../intersect/records/for_each_cut_chord.hpp"

#include <cstddef>
#include <utility>

namespace tf::iso {

/// CORE. One cut face's triangulation input. The boundary chain is the face's
/// corners with the field's hits in parametric order along each edge, lifted to
/// the flat identity space, and it names each identity exactly once — so the
/// chain IS the point table, its position IS the local index, and its edges are
/// consecutive positions. The face's interior chords complete the constraint
/// set, and the chain's own winding is stated here, where the projection is
/// already in hand.
///
/// A chord lying along a mesh edge cuts nothing, and marking one wall twice
/// would toggle its region separation, so boundary chords are dropped here.
template <typename Index, typename Int, typename Records, typename Face,
          typename FieldPoints, typename Local, typename PointOfFlat>
auto prepare_iso_face_cut(const Records &records, const Face &face,
                          Index n_original, const FieldPoints &field_points,
                          const std::pair<int, int> &axes, Local &local,
                          const PointOfFlat &point_of_flat) -> void {
  using T2 = typename tf::exact::meta<Int>::T2;

  const auto get_point = [&](Index tag, Index id) {
    return point_of_flat(tag == Index(-1) ? n_original + id : id);
  };
  local.loop.clear();
  // the field pipeline states a face's claims on one carrier, so the
  // loop's second one is empty
  tf::intersect::graph::extract_loop(
      records, tf::make_range(records.begin(), records.begin()), face,
      Index(0), get_point, local.loop_work, local.loop);

  const auto n_chain = local.loop.size();
  local.chain.allocate(n_chain);
  local.pts2.clear();
  local.pts2.reserve(n_chain);
  for (std::size_t i = 0; i < n_chain; ++i) {
    const auto &v = local.loop[i];
    const auto flat = v.source == tf::intersect::graph::vertex_source::created
                          ? n_original + v.id
                          : v.id;
    local.chain[i] = flat;
    const auto q = point_of_flat(flat);
    local.pts2.push_back(tf::exact::pt2<Int>{q[axes.first], q[axes.second]});
  }
  // a chain of no area declines the one-chord split, so the winding is
  // three-valued here and the sign, not the side, is the answer
  const auto area2 = tf::exact::signed_area_2x(
      tf::make_sequence_range(n_chain),
      [&](std::size_t i) { return local.pts2[i]; });
  local.chain_orientation = area2 > T2(0) ? 1 : area2 < T2(0) ? -1 : 0;

  // Every chord endpoint is a chain member, and a chain is a handful of
  // entries, so its own scan answers faster than a table beside it.
  const auto position_of = [&](Index flat) -> Index {
    Index position = 0;
    while (local.chain[std::size_t(position)] != flat)
      ++position;
    return position;
  };

  local.cons.clear();
  for (std::size_t i = 0; i < n_chain; ++i) {
    local.cons.push_back(Index(i));
    local.cons.push_back(Index(i + 1 == n_chain ? 0 : i + 1));
  }
  tf::intersect::for_each_cut_group(
      records, Index(face.size()), field_points,
      [&](Index id0, Index id1, bool on_boundary) {
        if (on_boundary)
          return;
        local.cons.push_back(position_of(n_original + id0));
        local.cons.push_back(position_of(n_original + id1));
      });
}

} // namespace tf::iso
