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
#include <array>
#include <cstddef>

namespace tf::arrangement::reseat {

/// Per tag its first piece name and per piece its rank among its tag's
/// pieces in the order they were stated, from each piece's `(tag, source)`:
/// the piece's name is `tag_base[tag] + rank`.
template <typename Index>
auto rank_pieces_by_tag(const tf::buffer<std::array<Index, 2>> &sources,
                        Index n_tags, tf::buffer<Index> &tag_base,
                        tf::buffer<Index> &rank) -> void {
  tag_base.allocate_and_initialize(std::size_t(n_tags) + 1, Index(0));
  rank.allocate(sources.size());
  for (std::size_t k = 0; k < sources.size(); ++k)
    rank[k] = tag_base[std::size_t(sources[k][0]) + 1]++;
  for (std::size_t tag = 1; tag < tag_base.size(); ++tag)
    tag_base[tag] += tag_base[tag - 1];
}

} // namespace tf::arrangement::reseat
