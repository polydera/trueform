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

#include "../../core/range.hpp"
#include "../../topology/triangulation/for_each_convex_chain_fan_triangle.hpp"
#include "./iso_band_of_triangles.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace tf::iso {

/// CORE. The state a scalar field cut almost always leaves: ONE chord, both
/// ends on the boundary chain. The chord separates the chain into two runs,
/// and those runs closed by the chord ARE the two pieces — no triangulation
/// has to discover them. The chain is the point table, so the chord already
/// arrives as two chain positions and the split is a walk.
///
/// The fan is the whole triangulation only if a piece is convex. A chain lies
/// on the face's own boundary, so a triangle face's chain is convex by
/// construction; every other arity, a degenerate projection, and a chord that
/// bounds no area decline, and the constrained build answers for them.
///
/// Returns whether this face was answered here.
template <typename LabelType, typename Index, typename Local,
          typename GetCategory, typename GetOnCut, typename GetCreatedCut>
auto split_iso_face_chord(Local &local, Index object, Index n_original,
                          std::size_t face_size,
                          const GetCategory &get_category,
                          const GetOnCut &get_on_cut,
                          const GetCreatedCut &get_created_cut) -> bool {
  const auto n_chain = local.chain.size();
  if (face_size != 3 || local.chain_orientation == 0 ||
      local.cons.size() != 2 * n_chain + 2)
    return false;

  auto position0 = std::size_t(local.cons[2 * n_chain]);
  auto position1 = std::size_t(local.cons[2 * n_chain + 1]);
  if (position0 > position1)
    std::swap(position0, position1);
  const auto near_side = position1 - position0 + 1;
  const auto far_side = n_chain - position1 + position0 + 1;
  if (near_side < 3 || far_side < 3)
    return false;

  const auto emit_piece = [&](std::size_t start, std::size_t count) {
    const auto first = local.tris.size();
    local.tri_offsets.push_back(Index(first));
    tf::topology::for_each_convex_chain_fan_triangle(
        local.chain, start, count,
        [&local](Index apex, Index a, Index b) {
          local.tris.push_back({apex, a, b});
        });
    local.bands.push_back(iso_band_of_triangles<LabelType>(
        tf::make_range(local.tris.begin() + std::ptrdiff_t(first),
                       local.tris.end()),
        n_original, get_category, get_on_cut, get_created_cut));
    local.faces.push_back(object);
  };

  emit_piece(position0, near_side);
  emit_piece(position1, far_side);
  return true;
}

} // namespace tf::iso
