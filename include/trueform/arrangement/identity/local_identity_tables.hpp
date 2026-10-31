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

#include "../../core/buffer.hpp"
#include "../../exact/canonical_plane.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/plane_identity_names.hpp"

#include <array>

namespace tf::arrangement {

/// The identities the local arrangement states: the name statements and the
/// classes they close into, every created identity's class, the closed
/// rewrite table, and the name of every plane it holds. Created id `id` of
/// the world indexes these past the intersections' own points. The layer `L`
/// of @ref identity_substrate, which alone holds it.
template <typename Index, typename Int> struct local_identity_tables {
  using param_t = typename tf::exact::meta<Int>::param_type;
  using pt3_t = tf::exact::pt3<Int>;

  /// The statements, sorted by name: a name's run is its class.
  tf::buffer<tf::intersect::graph::plane_name_statement<Index>> statements;
  tf::buffer<Index> class_offsets;
  tf::buffer<Index> class_id;
  // a created point's complete statement: its name (in `statements`) plus
  // the root it was born on and its dyadic parameter there. No coordinate
  // is stored — blending commutes with projection, so a consumer that
  // blends on demand reads exactly what a table would have held.
  tf::buffer<Index> class_root;
  tf::buffer<param_t> class_t;
  tf::buffer<Index> created_class;
  // a created identity a carve stated at a lattice point no name holds:
  // its class entry is `-(k + 1)`, the plane tier's own direct form
  tf::buffer<pt3_t> direct_points;
  /// `{source, id, target}` — identities the gate absorbed, in the rewrite
  /// currency the pipeline already applies, so a consumer holding an old id
  /// stays merge-blind.
  tf::buffer<std::array<Index, 3>> merges;
  /// The classes the split walk reads; a class minted to name a position
  /// lies past them.
  Index n_classes = 0;
  /// Per plane of the world, the name its founder stated (@ref
  /// tf::intersect::graph::name_plane_carriers): the cut planes, then those
  /// the entrants and the carves founded.
  tf::buffer<tf::exact::canonical_plane<Int>> plane_names;

  auto n_created() const -> Index { return Index(created_class.size()); }

  auto clear() -> void {
    statements.clear();
    class_offsets.clear();
    class_id.clear();
    class_root.clear();
    class_t.clear();
    created_class.clear();
    direct_points.clear();
    merges.clear();
    n_classes = 0;
    plane_names.clear();
  }
};

} // namespace tf::arrangement
