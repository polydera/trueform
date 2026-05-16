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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include <array>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Append one fragment-side merge per fragment carrying at
/// least one boundary edge to the provided merge buffer.
///
/// A manifold-edge component with a boundary edge does not separate
/// two volumes — its two sides are the same volume topologically,
/// joined around the boundary. The pair `{2*frag+0, 2*frag+1}`
/// realises that. Closed components have no boundary edges and
/// contribute nothing.
///
/// @param polygons Polygons range tagged with a manifold-edge link.
/// @param fragment_labels MEL-component labels per face.
/// @param merges Merge-pair buffer to append to. Caller is responsible
///        for sort + unique afterwards if combining with other merge
///        sources.
template <typename Polygons, typename FragLabels, typename Index>
void emit_boundary_merges(const Polygons &polygons,
                          const FragLabels &fragment_labels,
                          tf::buffer<std::array<Index, 2>> &merges) {
  tf::buffer<char> has_boundary;
  has_boundary.allocate(fragment_labels.n_components);
  tf::parallel_fill(has_boundary, char(0));

  tf::parallel_for_each(
      tf::enumerate(polygons.manifold_edge_link()), [&](auto pair) {
        auto &&[f, links] = pair;
        for (auto link : links) {
          if (link.is_boundary()) {
            has_boundary[fragment_labels.labels[f]] = char(1);
            return;
          }
        }
      });

  for (Index k = 0; k < Index(fragment_labels.n_components); ++k) {
    if (has_boundary[k])
      merges.push_back({Index(2 * k), Index(2 * k + 1)});
  }
}

} // namespace tf::topology::domains
