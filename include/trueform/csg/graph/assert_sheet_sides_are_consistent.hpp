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
#include "./arrangement_descriptor.hpp"
#include "./component_crossings.hpp"
#include "./domain_inclusions.hpp"
#include "./sheet_sides.hpp"
#include <cassert>
#include <cstddef>

namespace tf::csg::graph {

/// A sheet column holds a side only where one is stated: a domain without
/// one reads zero. Between two domains that both hold a side, a wall that
/// carries the sheet separates them, and a wall that does not keeps their
/// side unless the walls state both of them — two stated domains meeting
/// at the sheet's rim.
template <typename Index>
auto assert_sheet_sides_are_consistent(
    const tf::csg::graph::domain_inclusions &inc,
    const tf::csg::graph::sheet_sides<Index> &sides,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const tf::csg::graph::component_crossings<Index> &crossings) -> void {
  using sides_t = tf::csg::graph::sheet_sides<Index>;
  const Index n_sheets = sides.n_sheets();
  const Index n_domains = desc.n_domains;
  const Index n_components = Index(desc.tag_of_component.size());
  bool published = true;
  for (Index k = Index(0); k < n_sheets; ++k)
    for (Index d = Index(0); d < n_domains; ++d)
      published = published && inc.test(std::size_t(d),
                                         std::size_t(sides.tags[k])) ==
                                   (sides.state(k, d) == sides_t::behind);

  tf::buffer<char> walled;
  walled.allocate(sides.states.size());
  for (auto &w : walled)
    w = char(0);
  for (Index c = Index(0); c < n_components; ++c)
    sides.for_each_sheet_layer(desc, crossings, c, [&](Index k, Index) {
      for (Index s = Index(0); s < Index(2); ++s)
        walled[std::size_t(k) * std::size_t(n_domains) +
               std::size_t(desc.domain_of_side[2 * c + s])] = char(1);
    });

  bool consistent = true;
  for (Index c = Index(0); c < n_components; ++c) {
    const Index d0 = desc.domain_of_side[2 * c + 0];
    const Index d1 = desc.domain_of_side[2 * c + 1];
    if (d0 == d1)
      continue;
    for (Index k = Index(0); k < n_sheets; ++k) {
      const char s0 = sides.state(k, d0);
      const char s1 = sides.state(k, d1);
      const auto held = [](char s) {
        return s == sides_t::front || s == sides_t::behind;
      };
      if (!held(s0) || !held(s1))
        continue;
      if (sides.carries(desc, crossings, c, k)) {
        consistent = consistent && s0 != s1;
        continue;
      }
      const bool rim =
          walled[std::size_t(k) * std::size_t(n_domains) + std::size_t(d0)] &&
          walled[std::size_t(k) * std::size_t(n_domains) + std::size_t(d1)];
      consistent = consistent && (rim || s0 == s1);
    }
  }
  assert(published && "a sheet column reads a side nothing stated");
  assert(consistent && "a sheet column holds across a wall the sheet is on, "
                       "or changes across one it is not");
  (void)published;
  (void)consistent;
}

} // namespace tf::csg::graph
