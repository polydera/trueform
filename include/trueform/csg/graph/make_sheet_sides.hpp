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
#include "../../core/buffer.hpp"
#include "./arrangement_descriptor.hpp"
#include "./component_crossings.hpp"
#include "./sheet_sides.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The sheet sides the walls state — the local side of
///        @ref tf::csg::graph::sheet_sides — every other pair unstated.
///        Without sheets the table holds no state at all.
template <typename Index>
auto make_sheet_sides(const tf::csg::graph::arrangement_descriptor<Index> &desc,
                      const tf::csg::graph::component_crossings<Index> &crossings,
                      Index n_tags, const tf::buffer<char> &is_sheet_tag)
    -> sheet_sides<Index> {
  using sides_t = sheet_sides<Index>;
  sides_t out;
  out.n_domains = desc.n_domains;
  out.sheet_of_tag.allocate(static_cast<std::size_t>(n_tags));
  for (Index t = Index(0); t < n_tags; ++t) {
    const bool sheet = t < Index(is_sheet_tag.size()) && is_sheet_tag[t];
    out.sheet_of_tag[t] = sheet ? Index(out.tags.size()) : Index(-1);
    if (sheet)
      out.tags.push_back(t);
  }
  if (out.tags.size() == 0)
    return out;
  out.states.allocate(out.tags.size() *
                      static_cast<std::size_t>(desc.n_domains));
  tf::parallel_fill(out.states, sides_t::unstated);

  auto state = [&](Index k, Index d, char side) {
    char &s = out.state(k, d);
    s = s == sides_t::unstated || s == side ? side : sides_t::wraps;
  };
  const Index n_components = Index(desc.tag_of_component.size());
  for (Index c = Index(0); c < n_components; ++c) {
    const Index d0 = desc.domain_of_side[2 * c + 0];
    const Index d1 = desc.domain_of_side[2 * c + 1];
    out.for_each_sheet_layer(desc, crossings, c, [&](Index k, Index behind) {
      state(k, d0, behind == Index(0) ? sides_t::behind : sides_t::front);
      state(k, d1, behind == Index(1) ? sides_t::behind : sides_t::front);
    });
  }
  return out;
}

} // namespace tf::csg::graph
