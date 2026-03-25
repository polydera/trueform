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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/point.hpp"
#include "../../core/small_vector.hpp"
#include "../../topology/ear_cutter.hpp"
#include "../arrangement_map_data.hpp"

namespace tf::cut {

/// Triangulate cut face loops and produce triangle indices + labels.
///
/// For each cut face loop, projects to 2D using make_projector (int32),
/// triangulates with exact ear_cutter, and maps vertex indices through
/// map_data.map_vertex.
/// Produces three parallel buffers: triangle indices, tag labels, face labels.
template <typename Index, typename Range, typename MakeProjector>
auto triangulate_arrangement_cuts(
    const Range &zipped_descs_loops,
    const MakeProjector &make_projector,
    const tf::cut::arrangement_map_data<Index> &map_data,
    tf::buffer<Index> &triangles,
    tf::buffer<Index> &tag_labels,
    tf::buffer<Index> &face_labels) {

  tf::generic_generate(
      zipped_descs_loops,
      std::tie(triangles, tag_labels, face_labels),
      std::make_pair(tf::small_vector<tf::point<int32_t, 2>, 10>{},
                     tf::ear_cutter<Index>{}),
      [&make_projector, &map_data](
          const auto &pair, auto &buffers, auto &state) {
        auto [desc, loop] = pair;
        auto &[tri_buf, tag_buf, face_buf] = buffers;
        auto &[pts, earcut] = state;
        auto projector = make_projector(desc);
        pts.clear();
        for (const auto &v : loop)
          pts.push_back(projector(v));
        earcut.build(tf::make_points(pts));
        auto n_tris = earcut.indices_buffer().size() / 3;
        for (auto id : earcut.indices_buffer())
          tri_buf.push_back(map_data.map_vertex(desc.tag, loop[id]));
        for (std::size_t i = 0; i < n_tris; ++i) {
          tag_buf.push_back(desc.tag);
          face_buf.push_back(desc.object);
        }
      });
}

} // namespace tf::cut
