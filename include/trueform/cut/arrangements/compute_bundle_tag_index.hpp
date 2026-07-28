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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./component_labels.hpp"
#include "../face_regions.hpp"
#include "tbb/parallel_sort.h"
#include "tbb/task_group.h"
#include <algorithm>
#include <array>
#include <utility>

namespace tf::cut {

/// @ingroup cut
/// @brief Build `(tag_of_component, bundle_to_tags)`.
///
/// `tag_of_component[c]` is the form tag of component `c` (or
/// `Index(-1)` for dropped/empty). Single-valued under the per-tag CCL
/// invariant in `component_labels::_compute_loop_component_labels`.
///
/// `bundle_to_tags[b]` is the ascending-sorted unique list of form tags
/// whose components belong to bundle `b`. Suitable for binary-search
/// lookups.
template <typename Index, typename Int>
auto compute_bundle_tag_index(const tf::cut::component_labels<Index> &ag,
                              const tf::face_regions<Index, Int> &fr,
                              const tf::buffer<Index> &bundle_of_component,
                              Index n_bundles)
    -> std::pair<tf::buffer<Index>, tf::offset_block_buffer<Index, Index>> {
  using ag_t = tf::cut::component_labels<Index>;
  const Index n_components = ag.n_components();

  std::pair<tf::buffer<Index>, tf::offset_block_buffer<Index, Index>> result;
  auto &tag_of_component = result.first;
  auto &bundle_to_tags = result.second;

  tag_of_component.allocate(static_cast<std::size_t>(n_components));
  tf::parallel_fill(tag_of_component, Index(-1));

  if (n_components == 0)
    return result;

  // Fill tag_of_component from both surface sources. Concurrent writers
  // store the same value (per-tag CCL invariant), so the race is benign.
  auto descs = fr.descriptors();
  auto loop_labels = ag.loop_labels();
  const Index n_tags = ag.n_tags();

  tbb::task_group tg;
  tg.run([&] {
    tf::parallel_for_each(
        tf::make_sequence_range(static_cast<Index>(fr.loops().size())),
        [&](Index l) {
          const Index c = loop_labels[l];
          if (c == ag_t::none_label)
            return;
          tag_of_component[c] = descs[l].tag;
        });
  });
  for (Index t = Index(0); t < n_tags; ++t) {
    tg.run([&, t] {
      auto labels = ag.polygon_labels(t);
      tf::parallel_for_each(tf::enumerate(labels), [&, t](auto pair) {
        auto &&[f, c] = pair;
        (void)f;
        if (c == ag_t::none_label)
          return;
        tag_of_component[c] = t;
      });
    });
  }
  tg.wait();

  // Lex-sorted unique (bundle, tag) pairs → bundle-grouped ascending
  // tag runs; populate offset_block_buffer in place.
  tf::buffer<std::array<Index, 2>> pairs;
  tf::generic_generate(
      tf::make_sequence_range(n_components), pairs,
      [&](Index c, tf::buffer<std::array<Index, 2>> &out) {
        const Index t = tag_of_component[c];
        if (t == Index(-1))
          return;
        out.push_back({bundle_of_component[c], t});
      });

  tbb::parallel_sort(pairs.begin(), pairs.end());
  pairs.erase_till_end(std::unique(pairs.begin(), pairs.end()));

  auto &offsets = bundle_to_tags.offsets_buffer();
  auto &data = bundle_to_tags.data_buffer();

  offsets.allocate(static_cast<std::size_t>(n_bundles + 1));
  tf::parallel_fill(offsets, Index(0));
  for (const auto &p : pairs)
    offsets[p[0] + 1]++;
  for (Index b = Index(0); b < n_bundles; ++b)
    offsets[b + 1] += offsets[b];

  data.allocate(pairs.size());
  tf::parallel_for_each(
      tf::enumerate(pairs),
      [&](auto t) {
        auto &&[i, p] = t;
        data[i] = p[1];
      },
      tf::checked);

  return result;
}

} // namespace tf::cut
