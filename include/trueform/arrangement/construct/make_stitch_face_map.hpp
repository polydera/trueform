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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/stitch_index_map.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/take.hpp"
#include "../mesh_arrangement_index_map.hpp"
#include <cstddef>

namespace tf::arrangement {

/// Fill a stitch map's face axis and counts from an arrangement index map.
///
/// The point axis is left to the caller, which can move it in when the
/// arrangement map is transient.
///
/// The arrangement map carries the face axis as an inverse; the forward
/// direction is what a consumer holding an input face id needs, so it is
/// scattered here from the per-tag uncut ranges. Everything outside those
/// ranges has no input counterpart: they are disjoint and ascending under
/// either emission order, so the complement is the gaps plus the tail.
template <typename Index, typename Iter, std::size_t N>
auto make_stitch_face_map(const tf::mesh_arrangement_index_map<Index> &im,
                           tf::range<Iter, N> n_input_faces,
                           tf::stitch_index_map<Index> &out) -> void {
  const Index n_sources = im.n_tags;
  const Index n_faces = static_cast<Index>(im.face_labels.size());
  out.n_sources = n_sources;
  out.n_output_points = im.n_output_points;
  out.n_output_faces = n_faces;

  out.face_f.offsets_buffer().allocate(static_cast<std::size_t>(n_sources) + 1);
  out.face_f.offsets_buffer()[0] = 0;
  for (Index s = 0; s < n_sources; ++s)
    out.face_f.offsets_buffer()[s + 1] =
        out.face_f.offsets_buffer()[s] + static_cast<Index>(n_input_faces[s]);
  out.face_f.data_buffer().allocate(
      static_cast<std::size_t>(out.face_f.offsets_buffer().back()));
  tf::parallel_fill(out.face_f.data_buffer(), n_faces);

  const Index *offsets = out.face_f.offsets_buffer().data();
  Index *forward = out.face_f.data_buffer().data();
  for (Index s = 0; s < n_sources; ++s) {
    const Index base = offsets[s];
    tf::parallel_for_each(
        tf::make_sequence_range(im.uncut_faces[s][0], im.uncut_faces[s][1]),
        [&, base](Index f) { forward[base + im.face_labels[f]] = f; },
        tf::checked);
  }

  Index n_kept = 0;
  for (Index s = 0; s < n_sources; ++s)
    n_kept += im.uncut_faces[s][1] - im.uncut_faces[s][0];
  out.new_faces.allocate(static_cast<std::size_t>(n_faces - n_kept));

  Index written = 0;
  auto emit_gap = [&](Index begin, Index end) {
    if (end > begin) {
      tf::parallel_iota(
          tf::take(tf::drop(out.new_faces, written), std::size_t(end - begin)),
          begin);
      written += end - begin;
    }
  };
  Index at = 0;
  for (Index s = 0; s < n_sources; ++s) {
    emit_gap(at, im.uncut_faces[s][0]);
    at = im.uncut_faces[s][1];
  }
  emit_gap(at, n_faces);
}

} // namespace tf::arrangement
