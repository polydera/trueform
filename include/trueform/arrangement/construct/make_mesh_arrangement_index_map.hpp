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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../mesh_arrangement_index_map.hpp"
#include "./arrangement_map_data.hpp"
#include "./make_arrangement_point_inverse.hpp"

namespace tf::arrangement {

/// Finalize a (discarded) arrangement_map_data + the construct's tag_labels/
/// face_labels into the public N-mesh index map.
///
/// point_f (forward): original_map / point_offsets are already an offset-block
/// layout, so we bake each mesh's output offset into the values in place
/// (local -> global) and move both buffers in -- no per-mesh allocation. The
/// bake is one flat parallel pass over all entries (tag via a search on the
/// sorted point_offsets), so it stays balanced when mesh sizes are skewed.
///
/// point inverse (point_labels / point_tag_labels): scattered output-indexed
/// with an `end`-sentinel tail for created points.
///
/// Takes the shared @ref arrangement_point_map_data base — it carries only
/// the point axis; the face axis arrives as the labels and the uncut
/// ranges.
/// `uncut_faces` is the per-tag `[begin, end)` of the output faces that
/// survived uncut, in whatever order the producer emitted them.
template <typename Index>
auto make_mesh_arrangement_index_map(arrangement_point_map_data<Index> &&d,
                                     tf::buffer<Index> &&tag_labels,
                                     tf::buffer<Index> &&face_labels,
                                     tf::blocked_buffer<Index, 2> &&uncut_faces,
                                     Index n_output_points)
    -> tf::mesh_arrangement_index_map<Index> {
  tf::mesh_arrangement_index_map<Index> out;
  const Index n = d.n_meshes;
  out.n_original_points = d.total_original_points;
  out.n_tags = n;
  out.n_output_points = n_output_points;
  out.face_tag_labels = std::move(tag_labels);
  out.face_labels = std::move(face_labels);
  out.uncut_faces = std::move(uncut_faces);

  // point inverse, output-indexed, with the per-axis end-sentinel tail for
  // created points (tag ends at n_tags, id at n_output_points).
  tf::arrangement::make_arrangement_point_inverse(d, n_output_points,
                                          out.point_tag_labels,
                                          out.point_labels);

  // forward: original_map values are local (0-based per mesh); original_offsets
  // is each mesh's global output start. point_offsets already blocks
  // original_map, so move both in, then bake local -> global by iterating the
  // blocks zipped with original_offsets.
  out.point_f.offsets_buffer() = std::move(d.point_offsets);
  out.point_f.data_buffer() = std::move(d.original_map);
  // unmapped originals carry the construct's sentinel (= input count); remap it
  // to the index-map end, and shift mapped locals to global by their offset.
  // Index the moved buffers directly per block (parallel over tags, then within
  // each block) -- no offset-block view temporaries.
  const Index in_end = static_cast<Index>(out.point_f.data_buffer().size());
  Index *data = out.point_f.data_buffer().data();
  const Index *offs = out.point_f.offsets_buffer().data();
  tf::parallel_for_each(
      tf::make_sequence_range(Index(0), n),
      [&](Index t) {
        const Index off = d.original_offsets[t];
        tf::parallel_for_each(
            tf::make_sequence_range(offs[t], offs[t + 1]),
            [&, off](Index i) {
              data[i] = (data[i] == in_end) ? n_output_points : data[i] + off;
            },
            tf::checked);
      },
      tf::checked);
  return out;
}

} // namespace tf::arrangement
