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
#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../exact/plane_support.hpp"
#include "./triangle_component_labels.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// Every face of every operand stands in the partition: an uncut face carries
/// its component, every triangle of a cut face carries one — a coplanar
/// duplicate its survivor's — and a face holding neither bounds no area on
/// the lattice. A failed plane states its holes, so it disarms the check.
template <typename Index, typename Int, typename Arrangement,
          typename ApplyToForm, typename ReadPoint>
auto assert_faces_are_named(
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const ApplyToForm &apply_to_form, const ReadPoint &read_point) -> void {
  if (arrangement.failed().size() != 0)
    return;
  const auto triangle_labels = labels.triangle_labels();
  const auto row_of = arrangement.row_of();
  const auto exposed_of_row = arrangement.exposed_of_row();
  const auto coplanar_of = arrangement.arrangement().coplanar_of();
  const auto coplanar = arrangement.arrangement().coplanar_descriptors();
  const auto slot_base = arrangement.global().face_slot_base();
  const auto label_of = [&](Index e) {
    const Index label = triangle_labels[e];
    const Index stack = coplanar_of[std::size_t(row_of[e])];
    if (label != labels.none_label || stack == Index(-1))
      return label;
    const Index survivor =
        exposed_of_row[std::size_t(coplanar[std::size_t(stack)].survivor)];
    return survivor == Index(-1) ? labels.none_label
                                 : triangle_labels[survivor];
  };
  bool named = true;
  for (Index t = Index(0); t < arrangement.n_tags() && named; ++t)
    apply_to_form(t, [&](const auto &tagged_form) {
      const auto face_labels = labels.polygon_labels(t);
      const Index base = Index(slot_base[std::size_t(t)]);
      named = !tf::parallel_contains(
          tf::enumerate(tagged_form.faces()),
          [&labels, &face_labels, &arrangement, &label_of, &read_point, base,
           t](const auto &pair) {
            const auto &[f, face] = pair;
            if (face_labels[Index(f)] != labels.none_label)
              return false;
            const auto span = arrangement.slot_range(base + Index(f));
            if (span[0] == span[1]) {
              tf::exact::plane_support<Int> support;
              for (const auto vertex : face)
                support.offer(read_point(int(t), Index(vertex)));
              return support.size == 3;
            }
            for (Index e = span[0]; e < span[1]; ++e)
              if (label_of(e) == labels.none_label)
                return true;
            return false;
          },
          tf::checked);
    });
  tf::core::verify(named, "a face of an operand stands in no domain");
}

} // namespace tf::csg::graph
