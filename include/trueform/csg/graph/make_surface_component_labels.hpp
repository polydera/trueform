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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../topology/label_connected_components.hpp"
#include "../../topology/make_applier.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"

namespace tf::csg::graph {

/// Label connected components of uncut surface faces.
/// Faces that appear in descriptors are masked out, as are the deleted
/// objects; remaining faces are flood-filled through manifold_edge_link.
template <typename Index, typename LabelType, typename Policy,
          typename Descriptors, typename DeletedObjects>
auto make_surface_component_labels(const tf::polygons<Policy> &polygons,
                                   const Descriptors &descriptors,
                                   const DeletedObjects &deleted_objects,
                                   Index expected_components = 2) {
  static_assert(tf::has_manifold_edge_link_policy<Policy>,
                "Use polygons | tf::tag(manifold_edge_link)");
  tf::buffer<char> mask;
  mask.allocate(polygons.faces().size());
  tf::connected_component_labels<LabelType> cl;
  cl.labels.allocate(mask.size());
  tf::parallel_fill(mask, true);
  tf::parallel_for_each(descriptors, [&](const auto &desc) {
    mask[desc.object] = false;
    cl.labels[desc.object] = -1;
  });
  tf::parallel_for_each(deleted_objects, [&](const Index &object) {
    mask[object] = false;
    cl.labels[object] = -1;
  });
  cl.n_components = tf::label_connected_components_masked(
      cl.labels, mask, tf::make_applier(polygons.manifold_edge_link()),
      expected_components);
  return cl;
}

} // namespace tf::csg::graph
