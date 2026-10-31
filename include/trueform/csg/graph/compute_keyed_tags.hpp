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
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Per key, the ascending unique form tags of the components
///        `key_of_component` groups under it, as one offset block per key;
///        a component whose tag is `-1` names none.
template <typename Index, typename KeyOfComponent>
auto compute_keyed_tags(const tf::buffer<Index> &tag_of_component,
                        const KeyOfComponent &key_of_component, Index n_keys)
    -> tf::offset_block_buffer<Index, Index> {
  tf::offset_block_buffer<Index, Index> keyed_tags;
  const Index n_components = Index(tag_of_component.size());

  tf::buffer<std::array<Index, 2>> pairs;
  tf::generic_generate(
      tf::make_sequence_range(n_components), pairs,
      [&](Index c, tf::buffer<std::array<Index, 2>> &out) {
        const Index t = tag_of_component[c];
        if (t == Index(-1))
          return;
        out.push_back({key_of_component[c], t});
      });

  tbb::parallel_sort(pairs.begin(), pairs.end());
  pairs.erase_till_end(std::unique(pairs.begin(), pairs.end()));

  auto &offsets = keyed_tags.offsets_buffer();
  auto &data = keyed_tags.data_buffer();

  offsets.allocate(static_cast<std::size_t>(n_keys + 1));
  tf::parallel_fill(offsets, Index(0));
  for (const auto &p : pairs)
    offsets[p[0] + 1]++;
  for (Index k = Index(0); k < n_keys; ++k)
    offsets[k + 1] += offsets[k];

  data.allocate(pairs.size());
  tf::parallel_for_each(
      tf::enumerate(pairs),
      [&data](auto t) {
        auto &&[i, p] = t;
        data[i] = p[1];
      },
      tf::checked);

  return keyed_tags;
}

} // namespace tf::csg::graph
