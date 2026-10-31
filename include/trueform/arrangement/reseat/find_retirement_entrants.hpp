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
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../intersect/graph/plane_uncut_entrants.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement::reseat {

/// The source faces the carve's retirements reach that neither the world
/// (`world_face_of`) nor the carve names, as `{tag, face}` ascending: a
/// kind-0 weld retires an original, and every face holding it holds an
/// instance of it (@ref tf::intersect::graph::discover_weld_entrants). A
/// face the carve cut or entered answers its own retirement.
template <typename Index, typename World, typename FaceOf, typename Package,
          typename ApplyToForm, typename AlreadyNamed>
auto find_retirement_entrants(const World &world, const FaceOf &world_face_of,
                              const Package &package,
                              const ApplyToForm &apply_to_form,
                              const AlreadyNamed &already_named)
    -> tf::buffer<std::array<Index, 2>> {
  tf::buffer<std::array<Index, 2>> reached;
  tf::buffer<Index> retired;
  for (const auto &weld : package.welds)
    if (weld[0] == Index(0))
      retired.push_back(weld[1]);
  if (retired.size() == 0)
    return reached;
  std::sort(retired.begin(), retired.end());
  retired.erase_till_end(std::unique(retired.begin(), retired.end()));

  tf::buffer<std::array<Index, 2>> carved;
  for (const auto &source : package.piece_sources)
    carved.push_back(source);
  for (auto k = package.piece_sources.size();
       k < package.piece_descriptors.size(); ++k)
    carved.push_back({Index(package.piece_descriptors[k].tag),
                      package.piece_descriptors[k].object});
  std::sort(carved.begin(), carved.end());
  const auto is_named = [&](Index tag, Index object) {
    return world_face_of(tag, object) != Index(-1) ||
           already_named(tag, object) ||
           std::binary_search(carved.begin(), carved.end(),
                              std::array<Index, 2>{tag, object});
  };
  const auto face_offsets = world.face_offsets();
  tf::buffer<Index> keys;
  tf::intersect::graph::discover_weld_entrants(retired, world.vertex_offsets(),
                                               face_offsets, apply_to_form,
                                               is_named, keys);
  std::sort(keys.begin(), keys.end());
  keys.erase_till_end(std::unique(keys.begin(), keys.end()));
  for (const auto key : keys) {
    const auto tag = tf::exact::tag_of_flat_vertex(face_offsets, key);
    reached.push_back({Index(tag), key - face_offsets[std::size_t(tag)]});
  }
  return reached;
}

} // namespace tf::arrangement::reseat
