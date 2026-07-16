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
#include "../../core/checked.hpp"
#include "../../core/faces.hpp"
#include "../../core/polygons.hpp"
#include "../../core/static_size.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../domain_labels.hpp"
#include "../orient_faces_consistently.hpp"
#include <array>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Label a mesh whose duplicate-face stacks have been resolved
///        to survivors, then lift the labels back onto every face.
///
/// One copy per stack survives into the labeled submesh; dead copies
/// take the sentinel pair, like Mode-1 open fragments — they bound no
/// domain and no extraction emits them. The survivors materialize into
/// an owned face buffer and are re-oriented consistently: a stack's
/// copies oppose each other, so any per-face pick can mix sheets, and
/// the labeling's consistent-orientation precondition must be restored
/// rather than assumed. Flipped survivors lift their labels back with
/// the sides swapped.
///
/// `label` receives the survivor submesh (a `tf::polygons` over owned
/// faces and the input's points) and returns its
/// @ref tf::domain_labels.
template <typename Index, typename Polygons, typename Label>
auto resolve_face_stacks(const Polygons &polygons,
                         const tf::buffer<Index> &survivor_of,
                         const Label &label) -> tf::domain_labels<Index> {
  const Index n_faces = Index(polygons.size());
  tf::buffer<Index> pos_of;
  pos_of.allocate(polygons.size());
  tf::buffer<std::array<Index, 2>> pre_edge;
  constexpr auto N = tf::static_size_v<decltype(polygons.faces()[0])>;
  tf::buffer<Index> flat;
  tf::buffer<Index> offsets;
  Index n_survivors = 0;
  if constexpr (N == tf::dynamic_size)
    offsets.push_back(Index(0));
  for (Index f = 0; f < n_faces; ++f) {
    if (survivor_of[f] != f)
      continue;
    pos_of[f] = n_survivors++;
    auto face = polygons.faces()[f];
    pre_edge.push_back({Index(face[0]), Index(face[1])});
    for (auto v : face)
      flat.push_back(Index(v));
    if constexpr (N == tf::dynamic_size)
      offsets.push_back(Index(flat.size()));
  }
  auto finish = [&](auto sub_polygons) {
    tf::orient_faces_consistently(sub_polygons);
    auto sub = label(sub_polygons);
    tf::domain_labels<Index> out;
    out.n_domains = sub.n_domains;
    out.outer_shell_label = sub.outer_shell_label;
    out.labels.allocate(polygons.size());
    const Index sentinel = sub.n_domains;
    tf::parallel_for_each(
        tf::make_sequence_range(n_faces),
        [&](Index f) {
          auto dst = out.labels[f];
          if (survivor_of[f] != f) {
            dst[0] = sentinel;
            dst[1] = sentinel;
            return;
          }
          const Index k = pos_of[f];
          auto src = sub.labels[k];
          auto face = sub_polygons.faces()[k];
          const bool flipped = !(Index(face[0]) == pre_edge[k][0] &&
                                 Index(face[1]) == pre_edge[k][1]);
          dst[0] = flipped ? src[1] : src[0];
          dst[1] = flipped ? src[0] : src[1];
        },
        tf::checked);
    return out;
  };
  if constexpr (N == tf::dynamic_size)
    return finish(tf::make_polygons(
        tf::make_faces(tf::make_range(offsets), tf::make_range(flat)),
        polygons.points()));
  else
    return finish(tf::make_polygons(tf::make_faces<N>(tf::make_range(flat)),
                                    polygons.points()));
}

} // namespace tf::topology::domains
