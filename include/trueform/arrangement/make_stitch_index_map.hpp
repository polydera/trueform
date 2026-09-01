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

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/range.hpp"
#include "../core/stitch_index_map.hpp"
#include "./construct/make_stitch_face_map.hpp"
#include "./mesh_arrangement_index_map.hpp"
#include <cstddef>
#include <utility>

namespace tf {

/// @ingroup arrangement
/// @brief Read an arrangement result as a @ref tf::stitch_index_map, so the
///        operands' topology and trees can be carried onto it.
///
/// This is the only place that translates an arrangement into stitching terms.
/// Downstream, `tf::stitched_face_membership`,
/// `tf::stitched_manifold_edge_link` and `tf::stitch_mod_tree` know nothing
/// about tags, cuts, or emission order.
///
/// `n_input_faces` is the face count of each operand, in tag order.
template <typename Index, typename Iter, std::size_t N>
auto make_stitch_index_map(const tf::mesh_arrangement_index_map<Index> &im,
                           tf::range<Iter, N> n_input_faces)
    -> tf::stitch_index_map<Index> {
  tf::stitch_index_map<Index> out;
  tf::arrangement::make_stitch_face_map(im, n_input_faces, out);
  out.point_sources.allocate(im.point_tag_labels.size());
  out.point_ids.allocate(im.point_labels.size());
  tf::parallel_copy(im.point_tag_labels, out.point_sources);
  tf::parallel_copy(im.point_labels, out.point_ids);
  return out;
}

/// @ingroup arrangement
/// @brief Overload for an arrangement map that is not needed afterwards.
///
/// The point axis is the same fact under both names — a tag is a source, and
/// both mark their absent case with the axis end — so it moves across instead
/// of being copied.
/// @overload
template <typename Index, typename Iter, std::size_t N>
auto make_stitch_index_map(tf::mesh_arrangement_index_map<Index> &&im,
                           tf::range<Iter, N> n_input_faces)
    -> tf::stitch_index_map<Index> {
  tf::stitch_index_map<Index> out;
  tf::arrangement::make_stitch_face_map(im, n_input_faces, out);
  out.point_sources = std::move(im.point_tag_labels);
  out.point_ids = std::move(im.point_labels);
  return out;
}

} // namespace tf
