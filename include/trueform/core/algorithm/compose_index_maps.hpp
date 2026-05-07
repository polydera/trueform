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
#include "../index_map.hpp"
#include "../views/sequence_range.hpp"
#include "./parallel_for_each.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Compose two index maps `A -> B` and `B -> C` into `A -> C`.
///
/// Given a chain of mappings, produces the single index_map that goes
/// directly from the first space to the last:
/// - `f()[a]   = im2.f()[im1.f()[a]]`
/// - `kept_ids()[c] = im1.kept_ids()[im2.kept_ids()[c]]`
///
/// The resulting `kept_ids()` are expressed in `A`-space (the original
/// space), so the composed map looks identical to one produced directly
/// from `A` to `C`.
///
/// Removed entries follow the trueform sentinel convention: a forward
/// value of `f().size()` means the element was removed (matches
/// `tf::mask_to_index_map` and friends). The composed map propagates
/// the sentinel:
///   - if `im1.f()[a] == im1.f().size()` (removed in B), or
///   - if `im2.f()[b] == im2.f().size()` (removed in C),
/// then `out.f()[a] = out.f().size()` (removed in C), which equals
/// `im1.f().size()`.
///
/// @tparam Index The index type.
/// @param im1 First map: `A -> B`.
/// @param im2 Second map: `B -> C`.
/// @return Composed map: `A -> C`.
template <typename Index>
auto compose_index_maps(const tf::index_map_buffer<Index> &im1,
                        const tf::index_map_buffer<Index> &im2)
    -> tf::index_map_buffer<Index> {
  tf::index_map_buffer<Index> out;

  auto im1_sentinel = static_cast<Index>(im1.f().size());
  auto im2_sentinel = static_cast<Index>(im2.f().size());
  // out.f().size() == im1.f().size() so the out sentinel equals im1_sentinel.

  out.f().allocate(im1.f().size());
  tf::parallel_for_each(
      tf::make_sequence_range(static_cast<Index>(im1.f().size())),
      [&](Index a) {
        Index b = im1.f()[a];
        if (b == im1_sentinel) {
          out.f()[a] = im1_sentinel;
        } else {
          Index c = im2.f()[b];
          out.f()[a] = (c == im2_sentinel) ? im1_sentinel : c;
        }
      },
      tf::checked);

  out.kept_ids().allocate(im2.kept_ids().size());
  tf::parallel_for_each(
      tf::make_sequence_range(static_cast<Index>(im2.kept_ids().size())),
      [&](Index c) { out.kept_ids()[c] = im1.kept_ids()[im2.kept_ids()[c]]; },
      tf::checked);

  return out;
}

} // namespace tf
