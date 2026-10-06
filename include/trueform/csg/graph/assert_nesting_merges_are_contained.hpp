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
#include "../../core/aabb.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_descriptor.hpp"
#include "./compute_keyed_aabbs.hpp"
#include "./triangle_component_labels.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>

namespace tf::csg::graph {

/// A nesting merge lands a bundle inside its region, read off the cast's own
/// depths at the barrier that emitted the merges.
///
/// A bundle a closed volume of another bundle encloses — its depth on that
/// volume is nonzero — floats in a bounded region, and a bounded region lies
/// within the box of the walls naming it. So the bundle's box lies inside the
/// box of the walls naming the domain its envelope merged into. The check is
/// sound where it is armed and silent elsewhere: it cannot see a wrong
/// landing whose walls still cover the bundle's box, nor a bundle no single
/// closed volume encloses. It is a debug one — its caller states that — so
/// the boxes it reduces are built here and nothing keeps them.
template <typename Index, typename Int, typename Arrangement,
          typename ApplyToForm, typename ReadPoint>
auto assert_nesting_merges_are_contained(
    const tf::buffer<std::array<Index, 2>> &merges,
    const tf::buffer<Index> &outer_env,
    const tf::buffer<tf::aabb<Int, 3>> &bundle_boxes,
    const tf::buffer<Index> &seed_depth,
    const tf::buffer<char> &closed_volume_tags,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const ApplyToForm &apply_to_form, const ReadPoint &read_point) -> void {
  const auto n_tags = closed_volume_tags.size();
  const Index n_components = labels.n_components();
  auto walls_on_side = [&](Index s) {
    return tf::csg::graph::compute_keyed_aabbs<Index, Int>(
        tf::make_mapped_range(
            tf::make_sequence_range(n_components),
            [&desc, s](Index c) { return desc.domain_of_side[2 * c + s]; }),
        desc.n_domains, arrangement, labels, apply_to_form, read_point);
  };
  const auto walls_0 = walls_on_side(Index(0));
  const auto walls_1 = walls_on_side(Index(1));

  bool contained = true;
  for (Index b = Index(0); b < desc.n_bundles; ++b) {
    auto own_tags = desc.bundle_to_tags[b];
    bool enclosed = false;
    for (std::size_t t = 0; t < n_tags; ++t)
      enclosed = enclosed ||
                 (closed_volume_tags[t] &&
                  seed_depth[std::size_t(b) * n_tags + t] != Index(0) &&
                  !std::binary_search(own_tags.begin(), own_tags.end(),
                                      Index(t)));
    if (!enclosed)
      continue;
    const auto &box = bundle_boxes[b];
    for (const auto &merge : merges) {
      if (merge[0] != outer_env[b] && merge[1] != outer_env[b])
        continue;
      const Index d = merge[0] == outer_env[b] ? merge[1] : merge[0];
      for (int k = 0; k < 3; ++k)
        contained =
            contained &&
            std::min(walls_0[d].min[k], walls_1[d].min[k]) <= box.min[k] &&
            box.max[k] <= std::max(walls_0[d].max[k], walls_1[d].max[k]);
    }
  }
  assert(contained &&
         "a nesting merge lands a bundle outside its region's walls");
  (void)contained;
}

} // namespace tf::csg::graph
