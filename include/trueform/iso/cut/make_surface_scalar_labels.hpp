/*
 * Copyright (c) 2025 XLAB
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
#include "../../core/polygons.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/records/simple_intersections.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::iso {

/// Classify uncut surface faces by scalar band.
/// Cut faces get label -1. An uncut face takes its maximum vertex category:
/// a vertex sitting exactly on a cut value reports the band below, while
/// the face interior lies in the band above.
///
/// The -2 prefill is the unclassified state the band pass overwrites: the cut
/// pass writes only cut slots, so every other slot must already read as not -1.
template <typename LabelType, typename Index, typename Policy, typename RealT,
          std::size_t Dims, typename GetCategory>
auto make_surface_scalar_labels(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &si,
    const GetCategory &get_category) {
  tf::buffer<LabelType> out;
  out.allocate(polygons.size());
  tf::parallel_fill(out, LabelType(-2));
  tf::parallel_for_each(
      si.intersections(), [&](const auto &r) { out[r.front().object] = -1; },
      tf::checked);
  tf::parallel_for_each(
      tf::zip(out, polygons.faces()),
      [&](auto pair) {
        auto &&[label, face] = pair;
        if (label == -1)
          return;
        label = get_category(face[0]);
        for (std::size_t i = 1; i < std::size_t(face.size()); ++i)
          label = std::max(label, LabelType(get_category(face[i])));
      },
      tf::checked);
  return out;
}

} // namespace tf::iso
