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
#include "./domain_inclusions.hpp"
#include <array>
#include <cassert>
#include <cstddef>

namespace tf::csg::graph {

/// A nesting merge unites one region, read off the published inclusion bits.
///
/// A closed volume's bit is a winding number, a function of position, so it
/// is constant on a connected region, and a nesting merge unites two domains
/// of one region: their closed-volume bits agree. A sheet's bit is exempt — a
/// sheet that cuts a region only partially leaves one domain wrapping its
/// edge whose points carry both sides — and so is an open volume's, whose
/// crossing count depends on the path. A landing across a missed open wall
/// differs only in that wall's own column, which is exempt, so this check
/// cannot see one.
template <typename Index>
auto assert_nesting_merges_keep_closed_bits(
    const tf::buffer<std::array<Index, 2>> &merges,
    const tf::csg::graph::domain_inclusions &inc,
    const tf::buffer<char> &closed_volume_tags) -> void {
  bool kept = true;
  for (const auto &merge : merges)
    for (std::size_t t = 0; t < closed_volume_tags.size(); ++t)
      kept = kept && (!closed_volume_tags[t] ||
                      inc.test(std::size_t(merge[0]), t) ==
                          inc.test(std::size_t(merge[1]), t));
  assert(kept && "a nesting merge unites domains a closed volume separates");
  (void)kept;
}

} // namespace tf::csg::graph
