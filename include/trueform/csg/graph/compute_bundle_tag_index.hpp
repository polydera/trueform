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
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./compute_keyed_tags.hpp"
#include "tbb/task_group.h"
#include <cstddef>
#include <utility>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Build `(tag_of_component, bundle_to_tags)`.
///
/// `tag_of_component[c]` is the form tag of component `c` (or
/// `Index(-1)` for dropped/empty). Single-valued under the per-tag CCL
/// invariant the label tier establishes.
///
/// `bundle_to_tags[b]` is the ascending-sorted unique list of form tags
/// whose components belong to bundle `b`. Suitable for binary-search
/// lookups.
///
/// @param labels         The label tier: `n_components()`,
///                       `polygon_labels(t)`, `none_label`.
/// @param n_tags         The number of forms in the arrangement.
/// @param carrier_labels Per-carrier component labels of the cut
///                       surface; dead carriers hold `none_label`.
/// @param tags           Per-carrier form tags.
template <typename Index, typename Labels, typename CarrierLabels,
          typename Tags>
auto compute_bundle_tag_index(const Labels &labels, Index n_tags,
                              const CarrierLabels &carrier_labels,
                              const Tags &tags,
                              const tf::buffer<Index> &bundle_of_component,
                              Index n_bundles)
    -> std::pair<tf::buffer<Index>, tf::offset_block_buffer<Index, Index>> {
  const Index n_components = labels.n_components();

  std::pair<tf::buffer<Index>, tf::offset_block_buffer<Index, Index>> result;
  auto &tag_of_component = result.first;
  auto &bundle_to_tags = result.second;

  tag_of_component.allocate(static_cast<std::size_t>(n_components));
  tf::parallel_fill(tag_of_component, Index(-1));

  if (n_components == 0)
    return result;

  // Fill tag_of_component from both surface sources. Concurrent writers
  // store the same value (per-tag CCL invariant), so the race is benign.
  tbb::task_group tg;
  tg.run([&] {
    tf::parallel_for_each(
        tf::make_sequence_range(static_cast<Index>(carrier_labels.size())),
        [&](Index l) {
          const Index c = carrier_labels[l];
          if (c == Labels::none_label)
            return;
          tag_of_component[c] = tags[l];
        });
  });
  for (Index t = Index(0); t < n_tags; ++t) {
    tg.run([&, t] {
      auto face_labels = labels.polygon_labels(t);
      tf::parallel_for_each(
          tf::enumerate(face_labels), [&tag_of_component, t](auto pair) {
            auto &&[f, c] = pair;
            (void)f;
            if (c == Labels::none_label)
              return;
            tag_of_component[c] = t;
          });
    });
  }
  tg.wait();

  bundle_to_tags = tf::csg::graph::compute_keyed_tags(
      tag_of_component, bundle_of_component, n_bundles);

  return result;
}

} // namespace tf::csg::graph
