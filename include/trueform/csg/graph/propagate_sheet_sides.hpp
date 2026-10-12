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
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_descriptor.hpp"
#include "./assert_sheet_sides_are_consistent.hpp"
#include "./component_crossings.hpp"
#include "./domain_inclusions.hpp"
#include "./sheet_sides.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Carry every stated sheet side to the domains no wall states, and
///        publish the sheet columns of the inclusion table — the transit of
///        @ref tf::csg::graph::sheet_sides.
///
/// A seed — the winding's answer at a bundle's outer envelope,
/// `(domain, tag, behind, bundle)` — states a domain the walls and the
/// witnesses left unstated; where the envelope wraps the sheet, it states the
/// bundle's own domains instead. Seeds that disagree on one domain leave it
/// wrapping the sheet. Each sheet then floods on its own from every stated
/// domain.
template <typename Index>
auto propagate_sheet_sides(
    tf::csg::graph::domain_inclusions &inc,
    tf::csg::graph::sheet_sides<Index> &sides,
    const tf::buffer<std::array<Index, 4>> &sheet_seeds,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const tf::csg::graph::component_crossings<Index> &crossings,
    const tf::offset_block_buffer<Index, Index> &domain_components) -> void {
  using sides_t = tf::csg::graph::sheet_sides<Index>;
  const Index n_sheets = sides.n_sheets();
  const Index n_domains = desc.n_domains;
  if (n_sheets == Index(0) || n_domains == Index(0))
    return;

  tf::buffer<std::array<Index, 3>> stated; // (domain, tag, behind)
  tf::buffer<std::array<Index, 3>> moved;  // (bundle, tag, behind)
  for (const auto &seed : sheet_seeds) {
    if (sides.state(sides.sheet_of_tag[seed[1]], seed[0]) == sides_t::wraps)
      moved.push_back({seed[3], seed[1], seed[2]});
    else
      stated.push_back({seed[0], seed[1], seed[2]});
  }
  if (moved.size() > 0) {
    std::sort(moved.begin(), moved.end());
    const Index n_components = Index(desc.bundle_of_component.size());
    for (Index c = Index(0); c < n_components; ++c) {
      const Index b = desc.bundle_of_component[c];
      auto it = std::lower_bound(
          moved.begin(), moved.end(), b,
          [](const std::array<Index, 3> &m, Index key) { return m[0] < key; });
      for (; it != moved.end() && (*it)[0] == b; ++it)
        for (Index s = Index(0); s < Index(2); ++s)
          stated.push_back({desc.domain_of_side[2 * c + s], (*it)[1], (*it)[2]});
    }
  }
  tbb::parallel_sort(stated.begin(), stated.end(),
                     [](const auto &a, const auto &b) {
                       if (a[1] != b[1])
                         return a[1] < b[1];
                       return a[0] < b[0];
                     });
  for (std::size_t lo = 0; lo < stated.size();) {
    std::size_t hi = lo;
    char side = sides_t::unstated;
    while (hi < stated.size() && stated[hi][0] == stated[lo][0] &&
           stated[hi][1] == stated[lo][1]) {
      const char s = stated[hi][2] ? sides_t::behind : sides_t::front;
      side = side == sides_t::unstated || side == s ? s : sides_t::wraps;
      ++hi;
    }
    char &state = sides.state(sides.sheet_of_tag[stated[lo][1]], stated[lo][0]);
    if (state == sides_t::unstated)
      state = side;
    lo = hi;
  }

  tf::parallel_for_each(
      tf::make_sequence_range(n_sheets),
      [&](Index k, tf::buffer<Index> &queue) {
        queue.clear();
        for (Index d = Index(0); d < n_domains; ++d) {
          const char s = sides.state(k, d);
          if (s == sides_t::front || s == sides_t::behind)
            queue.push_back(d);
        }
        for (std::size_t head = 0; head < queue.size(); ++head) {
          const Index d = queue[head];
          const char s = sides.state(k, d);
          for (const Index c : domain_components[std::size_t(d)]) {
            const Index d0 = desc.domain_of_side[2 * c + 0];
            const Index other = d0 == d ? desc.domain_of_side[2 * c + 1] : d0;
            if (sides.state(k, other) != sides_t::unstated ||
                sides.carries(desc, crossings, c, k))
              continue;
            sides.state(k, other) = s;
            queue.push_back(other);
          }
        }
      },
      tf::buffer<Index>{});

  tf::parallel_for_each(
      tf::make_sequence_range(n_domains),
      [&](Index d) {
        for (Index k = Index(0); k < n_sheets; ++k) {
          if (sides.state(k, d) == sides_t::behind)
            inc.set(std::size_t(d), std::size_t(sides.tags[k]));
          else
            inc.clear(std::size_t(d), std::size_t(sides.tags[k]));
        }
      },
      tf::checked);
#ifndef NDEBUG
  tf::csg::graph::assert_sheet_sides_are_consistent(inc, sides, desc,
                                                    crossings);
#endif
}

} // namespace tf::csg::graph
