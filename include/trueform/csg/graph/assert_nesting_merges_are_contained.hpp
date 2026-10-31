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
#include "../../core/config/verify.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_descriptor.hpp"
#include "./compute_keyed_aabbs.hpp"
#include "./domain_inclusions.hpp"
#include "./make_domain_islands.hpp"
#include "./make_island_envelopes.hpp"
#include "./triangle_component_labels.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// A nesting merge lands an island inside its region, read off the published
/// inclusion bits of the island's outer env — the cast's depths, kept as the
/// flood's seeds.
///
/// An island a closed volume it does not hold encloses — its depth on that
/// volume is nonzero — floats in a bounded region, and a bounded region lies
/// within the box of the walls naming it. So the island's box lies inside the
/// box of the walls naming the domain its envelope merged into. The check is
/// sound where it is armed and silent elsewhere: it cannot see a wrong
/// landing whose walls still cover the island's box, nor an island no single
/// closed volume encloses. It is a verification one — its caller states
/// that — so the islands and boxes it reduces are built here and nothing
/// keeps them.
template <typename Index, typename Int, typename VolT, typename Arrangement,
          typename ApplyToForm, typename ReadPoint>
auto assert_nesting_merges_are_contained(
    const tf::buffer<std::array<Index, 2>> &merges,
    const tf::csg::graph::domain_inclusions &inc,
    const tf::buffer<char> &closed_volume_tags,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const tf::buffer<VolT> &domain_volumes, const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const ApplyToForm &apply_to_form, const ReadPoint &read_point) -> void {
  if (merges.size() == 0)
    return;
  const auto n_tags = closed_volume_tags.size();
  const Index n_components = labels.n_components();
  const auto islands = tf::csg::graph::make_domain_islands(desc);
  const bool bundled = islands.n_islands == desc.n_bundles;
  const auto &island_of_component =
      bundled ? desc.bundle_of_component : islands.island_of_component;
  const auto &island_to_tags =
      bundled ? desc.bundle_to_tags : islands.island_to_tags;
  const auto outer_env = tf::csg::graph::make_island_envelopes(
      island_of_component, islands.n_islands, desc, domain_volumes);
  const auto island_boxes = tf::csg::graph::compute_keyed_aabbs<Index, Int>(
      island_of_component, islands.n_islands, arrangement, labels,
      apply_to_form, read_point);
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
  for (Index i = Index(0); i < Index(outer_env.size()); ++i) {
    auto own_tags = island_to_tags[i];
    bool enclosed = false;
    for (std::size_t t = 0; t < n_tags; ++t)
      enclosed =
          enclosed ||
          (closed_volume_tags[t] &&
           inc.test(std::size_t(outer_env[i]), t) &&
           !std::binary_search(own_tags.begin(), own_tags.end(), Index(t)));
    if (!enclosed)
      continue;
    const auto &box = island_boxes[i];
    for (const auto &merge : merges) {
      if (merge[0] != outer_env[i] && merge[1] != outer_env[i])
        continue;
      const Index d = merge[0] == outer_env[i] ? merge[1] : merge[0];
      for (int k = 0; k < 3; ++k)
        contained =
            contained &&
            std::min(walls_0[d].min[k], walls_1[d].min[k]) <= box.min[k] &&
            box.max[k] <= std::max(walls_0[d].max[k], walls_1[d].max[k]);
    }
  }
  tf::core::verify(
      contained, "a nesting merge lands an island outside its region's walls");
}

} // namespace tf::csg::graph
