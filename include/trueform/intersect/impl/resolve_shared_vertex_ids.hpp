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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../exact/vertex.hpp"
#include "../exact/tagged_intersection.hpp"

#include <algorithm>

namespace tf::intersect {

/// Replace sentinel ids (>= base) with real point ids: collect the
/// referenced original vertices (slot = offset vertex id, decomposed
/// through `vertex_offsets` to a (tag, vertex) pair), append their
/// exact coordinates once, remap. Runs upstream of dedup, so a shared
/// vertex that coincides with any computed intersection point merges
/// canonically.
template <typename Index, typename Int, typename ApplyToForm,
          typename Converter>
auto resolve_shared_vertex_ids(
    tf::buffer<tf::intersect::tagged_intersection<Index>> &intersections,
    tf::buffer<tf::exact::pt3<Int>> &points, Index base,
    const ApplyToForm &apply_to_form, const tf::buffer<Index> &vertex_offsets,
    const Converter &converter) {
  using intersection_t = tf::intersect::tagged_intersection<Index>;
  tf::buffer<Index> slots;
  tf::generic_generate(intersections, slots,
                       [base](const intersection_t &rec, auto &out) {
                         if (rec.id >= base)
                           out.push_back(rec.id - base);
                       });
  if (slots.size() == 0)
    return;
  std::sort(slots.begin(), slots.end());
  slots.erase_till_end(std::unique(slots.begin(), slots.end()));
  auto n_new = slots.size();
  points.reallocate(std::size_t(base) + n_new);
  for (std::size_t k = 0; k < n_new; ++k) {
    auto slot = slots[k];
    auto tag = int(std::upper_bound(vertex_offsets.begin(),
                                    vertex_offsets.end(), slot) -
                   vertex_offsets.begin()) -
               1;
    auto vid = slot - vertex_offsets[std::size_t(tag)];
    apply_to_form(tag, [&](const auto &form) {
      points[std::size_t(base) + k] = converter.convert(form.points()[vid]);
    });
  }
  tf::parallel_for_each(intersections, [&](intersection_t &rec) {
    if (rec.id >= base) {
      auto it = std::lower_bound(slots.begin(), slots.end(), rec.id - base);
      rec.id = base + Index(it - slots.begin());
    }
  });
}

} // namespace tf::intersect
