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

#include "../core/blocked_buffer.hpp"
#include "../core/offset_block_buffer.hpp"
#include <cstdint>

namespace tf {

/// Relates the cells of @ref tf::make_csg_domains back to the N input forms.
///
/// A domain decomposition has one mesh per cell, each with its own
/// re-deduplicated points, so every axis is an @ref tf::offset_block_buffer
/// whose blocks run parallel to `cells`: block `k` is cell `k`. `face_*`
/// blocks have one entry per cell face, `point_*` blocks one per cell point.
///
/// There is no forward map: a shared boundary point or face belongs to several
/// cells, so "input element -> its cell element" is not single-valued; the map
/// is inverse-only.
///
/// Created intersection points have no input origin and carry the `end`
/// sentinel, each axis ending past its own range: the tag ends at `n_tags`,
/// the point id at `n_output_points`. Cell points are emitted in cut order,
/// not `[originals | created]`, so there is no positional boundary to test
/// against -- detect a created point by the **tag** sentinel,
/// `point_tag_blocks[k][p] == n_tags`. That check is always reliable because
/// valid tags are `[0, n_tags)`; the point-id sentinel (`n_output_points`)
/// is not a safe detector under a filter, where a surviving input-point id
/// can reach it.
template <typename Index> struct csg_domains_index_map {
  /// cell face -> input form tag
  tf::offset_block_buffer<Index, Index> face_tag_blocks;
  /// cell face -> input face id within its form
  tf::offset_block_buffer<Index, Index> face_blocks;
  /// cell point -> input form tag; created -> n_tags
  tf::offset_block_buffer<Index, Index> point_tag_blocks;
  /// cell point -> input point id within its form; created -> n_output_points
  tf::offset_block_buffer<Index, Index> point_blocks;
  /// global arrangement points below this are kept originals, at/above created
  Index n_original_points = 0;
  /// number of input forms; the tag axis `end` sentinel value
  Index n_tags = 0;
  /// total global arrangement points; the point-id axis `end` sentinel value
  Index n_output_points = 0;
  /// Per-cell operand inclusion, block size `n_tags`: `inclusion[k][i]`
  /// is true iff cell `k` lies inside form `i` (for a sheet: behind the
  /// sheet's normal).
  tf::blocked_buffer<bool, tf::dynamic_size> inclusion;
};

} // namespace tf
