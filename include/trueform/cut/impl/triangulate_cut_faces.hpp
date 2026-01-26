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
#include "../../geometry/impl/ear_cutter.hpp"

namespace tf::cut {
template <typename Range, typename F0, typename F1, typename Index>
auto triangulate_cut_faces(const Range &zipped_cut_faces,
                           const F0 &make_projector, const F1 &id_map,
                           tf::buffer<Index> &out) {
  tf::generic_generate(
      zipped_cut_faces, out,
      std::make_pair(tf::small_vector<tf::point<double, 2>, 10>{},
                     tf::geom::earcutter<Index>{}),
      [&make_projector, &id_map](const auto &pair, auto &buffer, auto &state) {
        auto [descriptor, loop] = pair;
        auto &[pts, earcut] = state;
        auto projector = make_projector(descriptor);
        pts.clear();
        for (const auto &v : loop)
          pts.push_back(projector(v));
        earcut(pts);
        for (auto id : earcut.indices())
          buffer.push_back(id_map(loop[id]));
      });
}
} // namespace tf::cut
