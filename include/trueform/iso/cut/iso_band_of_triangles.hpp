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

#include <algorithm>

namespace tf::iso {

/// CORE. The band one piece of a cut face lies in, and the only producer of
/// that fact for either state the cut splits into.
///
/// A piece lies strictly inside one band, so the first corner that is an
/// original off every cut value states it. A piece with none is bounded by
/// chords and on-cut corners alone, and takes the highest cut index bounding
/// it. A corner the triangulation created names no band and is skipped.
template <typename LabelType, typename Index, typename Triangles,
          typename GetCategory, typename GetOnCut, typename GetCreatedCut>
auto iso_band_of_triangles(const Triangles &triangles, Index n_original,
                           const GetCategory &get_category,
                           const GetOnCut &get_on_cut,
                           const GetCreatedCut &get_created_cut) -> LabelType {
  LabelType band = 0;
  for (const auto &triangle : triangles)
    for (const auto flat : triangle) {
      if (flat < Index(0))
        continue;
      if (flat >= n_original) {
        band = std::max(band, LabelType(get_created_cut(flat - n_original)));
        continue;
      }
      if (!get_on_cut(flat))
        return LabelType(get_category(flat));
      band = std::max(band, LabelType(get_category(flat)));
    }
  return band;
}

} // namespace tf::iso
