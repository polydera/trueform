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
#include "../core/index_map.hpp"
#include "../core/none.hpp"
#include "../core/points.hpp"
#include "../core/polygons.hpp"
#include "../core/segments.hpp"
#include "../core/stitch_index_maps.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "./mod_tree.hpp"

namespace tf::spatial {

/// Stitch mod_tree when it is the left operand of a boolean operation.
/// When right is none, kept0 primitives are remapped via polygons0.f().
/// Everything else (kept1 + dirty) is added as new, starting at polygons1_offset.
template <typename Index, typename BV, typename Primitives>
auto stitch_mod_tree(const Primitives &result_primitives,
                     tf::mod_tree<Index, BV> &mod_tree0, tf::none_t,
                     const tf::stitch_index_maps<Index> &im,
                     tf::tree_config config) -> void {
  const Index n_new =
      Index(result_primitives.size()) - im.polygons1_offset;

  // New primitives start at polygons1_offset (kept1 + dirty)
  auto new_ids = tf::make_mapped_range(
      tf::make_sequence_range(Index(0), n_new),
      [&](Index i) { return im.polygons1_offset + i; });

  auto keep_if = [](Index id) {
    constexpr Index sentinel = Index(-1);
    return id != sentinel;
  };

  auto index_map =
      tf::make_index_map(tf::make_range(im.polygons0.f()), new_ids);
  mod_tree0.update_tree(result_primitives, index_map, keep_if, config);
}

/// Stitch mod_tree when it is the right operand of a boolean operation.
/// When left is none, kept1 primitives are remapped via polygons1.f() + offset.
/// Everything else (kept0 + dirty) is added as new.
template <typename Index, typename BV, typename Primitives>
auto stitch_mod_tree(const Primitives &result_primitives, tf::none_t,
                     tf::mod_tree<Index, BV> &mod_tree1,
                     const tf::stitch_index_maps<Index> &im,
                     tf::tree_config config) -> void {
  const Index kept0 = im.polygons0.kept_ids().size();
  const Index kept1 = im.polygons1.kept_ids().size();
  const Index dirty_start = kept0 + kept1;
  const Index n_dirty = Index(result_primitives.size()) - dirty_start;

  // New primitives: kept0 at [0, kept0) + dirty at [dirty_start, end)
  auto new_ids = tf::make_mapped_range(
      tf::make_sequence_range(Index(0), kept0 + n_dirty), [&](Index i) {
        return i < kept0 ? i : dirty_start + (i - kept0);
      });

  auto keep_if = [](Index id) {
    constexpr Index sentinel = Index(-1);
    return id != sentinel;
  };

  // polygons1.f() is 0-based, need to add offset for actual result positions
  auto old_ids = tf::make_mapped_range(im.polygons1.f(), [&](Index id) {
    constexpr Index sentinel = Index(-1);
    return id != sentinel ? id + im.polygons1_offset : sentinel;
  });

  auto index_map = tf::make_index_map(tf::make_range(old_ids), new_ids);
  mod_tree1.update_tree(result_primitives, index_map, keep_if, config);
}

} // namespace tf::spatial

namespace tf {

/// Stitch mod_tree for polygons (left operand).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::polygons<Policy> &result,
                     tf::mod_tree<Index, BV> &mod_tree0, tf::none_t,
                     const tf::stitch_index_maps<Index> &im,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, mod_tree0, tf::none, im, config);
}

/// Stitch mod_tree for polygons (right operand).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::polygons<Policy> &result, tf::none_t,
                     tf::mod_tree<Index, BV> &mod_tree1,
                     const tf::stitch_index_maps<Index> &im,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, tf::none, mod_tree1, im, config);
}

/// Stitch mod_tree for points (left operand).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::points<Policy> &result,
                     tf::mod_tree<Index, BV> &mod_tree0, tf::none_t,
                     const tf::stitch_index_maps<Index> &im,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, mod_tree0, tf::none, im, config);
}

/// Stitch mod_tree for points (right operand).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::points<Policy> &result, tf::none_t,
                     tf::mod_tree<Index, BV> &mod_tree1,
                     const tf::stitch_index_maps<Index> &im,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, tf::none, mod_tree1, im, config);
}

/// Stitch mod_tree for segments (left operand).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::segments<Policy> &result,
                     tf::mod_tree<Index, BV> &mod_tree0, tf::none_t,
                     const tf::stitch_index_maps<Index> &im,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, mod_tree0, tf::none, im, config);
}

/// Stitch mod_tree for segments (right operand).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::segments<Policy> &result, tf::none_t,
                     tf::mod_tree<Index, BV> &mod_tree1,
                     const tf::stitch_index_maps<Index> &im,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, tf::none, mod_tree1, im, config);
}

} // namespace tf
