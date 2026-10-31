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
#include "../../core/aabb_from.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/transformed.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../spatial/aabb_from.hpp"
#include "../../spatial/policy/tree.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace tf::arrangement::dispatch {

/// THE INPUT QUANTUM, stated once per build and carried by the graph: one
/// unit in the last place of the largest coordinate magnitude of the box the
/// converter is built over — each operand's tree box in its frame — scaled
/// onto the lattice. Two identities closer than it are ones the input cannot
/// tell apart; an integral input is exact on its lattice and has none. The
/// box extent IS the quantum's law: it bounds every position a statement can
/// seat, created points included, so the quantum is never below the input
/// points' own ulp — a framed operand's box may state it one binade above,
/// and a threshold wider by that binade only refuses more degenerate seats.
template <typename Int, typename Real, typename ApplyToForm, typename Index>
auto make_input_quantum(
    const tf::exact::vertex_converter<Int, Real, 3> &converter,
    const ApplyToForm &apply_to_form, Index n_tags) -> Int {
  if constexpr (!std::is_floating_point_v<Real>) {
    (void)converter;
    (void)apply_to_form;
    (void)n_tags;
    return Int(0);
  } else {
    Real extent = 0;
    for (Index tag = 0; tag < n_tags; ++tag)
      apply_to_form(tag, [&](const auto &form) {
        using form_t = std::decay_t<decltype(form)>;
        const auto box = [&] {
          if constexpr (tf::has_tree_policy<form_t>)
            return tf::transformed(tf::aabb_from(form.tree()),
                                   tf::frame_of(form));
          else
            return tf::transformed(tf::aabb_from(form.points()),
                                   tf::frame_of(form));
        }();
        for (std::size_t k = 0; k < 3; ++k)
          extent = std::max({extent, Real(std::abs(Real(box.min[k]))),
                             Real(std::abs(Real(box.max[k])))});
      });
    const Real ulp =
        std::nextafter(extent, std::numeric_limits<Real>::infinity()) - extent;
    return Int(std::ceil(double(ulp) * double(converter.coords.scale)));
  }
}

} // namespace tf::arrangement::dispatch
