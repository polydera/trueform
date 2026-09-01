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
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/sequenced_generate.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/none.hpp"
#include "../core/points.hpp"
#include "../core/polygons.hpp"
#include "../core/range.hpp"
#include "../core/segments.hpp"
#include "../core/stitch_index_map.hpp"
#include "../core/views/sequence_range.hpp"
#include "./mod_tree.hpp"
#include "./tree_index_map.hpp"
#include <cstddef>

namespace tf::spatial {

/// Re-seat one source's mod_tree onto a merged mesh.
///
/// The tree keeps the primitives that survived whole, remapped through
/// `sim.face_f[source]`; everything it cannot inherit — the other sources'
/// primitives and every new face — enters the delta tree.
template <typename Index, typename BV, typename Primitives>
auto stitch_mod_tree(const Primitives &result_primitives,
                     tf::mod_tree<Index, BV> &tree, Index source,
                     const tf::stitch_index_map<Index> &sim,
                     tf::tree_config config) -> void {
  const Index n_faces = sim.n_output_faces;
  auto forward = sim.face_f[source];

  tf::buffer<bool> inherited;
  inherited.allocate(static_cast<std::size_t>(n_faces));
  tf::parallel_fill(inherited, false);
  tf::parallel_for_each(
      forward,
      [&](Index f) {
        if (f != n_faces)
          inherited[f] = true;
      },
      tf::checked);

  tf::buffer<Index> dirty_ids;
  tf::sequenced_generate(tf::make_sequence_range(Index(0), n_faces), dirty_ids,
                         [&](Index f, auto &out) {
                           if (!inherited[f])
                             out.push_back(f);
                         });

  // face_f's values are output ids, which can reach its own length, so the
  // removed marker has to be stated rather than derived from it.
  auto tree_map =
      tf::make_tree_index_map(forward, tf::make_range(dirty_ids), n_faces);
  tree.update(result_primitives, tree_map, config);
}

} // namespace tf::spatial

namespace tf {

/// @ingroup spatial_structures
/// @brief Stitch mod_tree for polygons (first source).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::polygons<Policy> &result,
                     tf::mod_tree<Index, BV> &mod_tree0, tf::none_t,
                     const tf::stitch_index_map<Index> &sim,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, mod_tree0, Index(0), sim, config);
}

/// @ingroup spatial_structures
/// @brief Stitch mod_tree for polygons (second source).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::polygons<Policy> &result, tf::none_t,
                     tf::mod_tree<Index, BV> &mod_tree1,
                     const tf::stitch_index_map<Index> &sim,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, mod_tree1, Index(1), sim, config);
}

/// @ingroup spatial_structures
/// @brief Stitch mod_tree for points (first source).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::points<Policy> &result,
                     tf::mod_tree<Index, BV> &mod_tree0, tf::none_t,
                     const tf::stitch_index_map<Index> &sim,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, mod_tree0, Index(0), sim, config);
}

/// @ingroup spatial_structures
/// @brief Stitch mod_tree for points (second source).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::points<Policy> &result, tf::none_t,
                     tf::mod_tree<Index, BV> &mod_tree1,
                     const tf::stitch_index_map<Index> &sim,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, mod_tree1, Index(1), sim, config);
}

/// @ingroup spatial_structures
/// @brief Stitch mod_tree for segments (first source).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::segments<Policy> &result,
                     tf::mod_tree<Index, BV> &mod_tree0, tf::none_t,
                     const tf::stitch_index_map<Index> &sim,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, mod_tree0, Index(0), sim, config);
}

/// @ingroup spatial_structures
/// @brief Stitch mod_tree for segments (second source).
template <typename Index, typename BV, typename Policy>
auto stitch_mod_tree(const tf::segments<Policy> &result, tf::none_t,
                     tf::mod_tree<Index, BV> &mod_tree1,
                     const tf::stitch_index_map<Index> &sim,
                     tf::tree_config config) -> void {
  tf::spatial::stitch_mod_tree(result, mod_tree1, Index(1), sim, config);
}

} // namespace tf
