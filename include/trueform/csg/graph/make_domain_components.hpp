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
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "./arrangement_descriptor.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Per domain, the components it bounds — a component holding the
///        domain on both of its sides is listed twice. The adjacency the
///        floods over the domains walk.
template <typename Index>
auto make_domain_components(
    const tf::csg::graph::arrangement_descriptor<Index> &desc)
    -> tf::offset_block_buffer<Index, Index> {
  const Index n_domains = desc.n_domains;
  const Index n_components = Index(desc.tag_of_component.size());
  tf::offset_block_buffer<Index, Index> out;
  auto &offsets = out.offsets_buffer();
  offsets.allocate(static_cast<std::size_t>(n_domains + 1));
  tf::parallel_fill(offsets, Index(0));
  for (Index c = 0; c < 2 * n_components; ++c)
    offsets[desc.domain_of_side[c] + 1]++;
  for (Index i = 0; i < n_domains; ++i)
    offsets[i + 1] += offsets[i];

  auto &data = out.data_buffer();
  data.allocate(static_cast<std::size_t>(offsets[n_domains]));
  tf::buffer<Index> cursor;
  cursor.allocate(static_cast<std::size_t>(n_domains));
  tf::parallel_for_each(
      tf::enumerate(cursor),
      [&offsets](auto pair) {
        auto &&[i, c] = pair;
        c = offsets[i];
      },
      tf::checked);
  for (Index c = 0; c < 2 * n_components; ++c)
    data[cursor[desc.domain_of_side[c]]++] = c / 2;
  return out;
}

} // namespace tf::csg::graph
