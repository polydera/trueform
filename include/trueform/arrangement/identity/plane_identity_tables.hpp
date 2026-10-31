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
#include "../planes/plane_recovery_name.hpp"

#include <array>

namespace tf::arrangement {

/// The identities the plane arrangement's waves state: every created
/// identity's class, the name and parameter that place it, the closed
/// rewrite table, and the name of every plane its promotion founded. Created
/// id `id` of the arrangement indexes these past the created extent its build
/// stood on, a founded plane `plane_names` past the plane extent. The layer
/// `P` of @ref identity_substrate, which alone holds it.
template <typename Index, typename Int> struct plane_identity_tables {
  using param_t = typename tf::exact::meta<Int>::param_type;
  using pt3_t = tf::exact::pt3<Int>;
  using name_t = plane_recovery_name<Index, param_t>;

  /// Closed, so one search states what any retired identity speaks now.
  /// Empty while nothing has been welded.
  tf::buffer<std::array<Index, 3>> merges;
  tf::buffer<Index> created_class;
  // plane-local refined points: exact direct forms, owner-aligned; a
  // negative `created_class` entry indexes them
  tf::buffer<pt3_t> direct_points;
  tf::buffer<param_t> class_t;
  tf::buffer<name_t> class_name;
  /// Per plane the promotion founded, the name its founding face stated.
  tf::buffer<tf::exact::canonical_plane<Int>> plane_names;

  auto n_created() const -> Index { return Index(created_class.size()); }

  auto clear() -> void {
    merges.clear();
    created_class.clear();
    direct_points.clear();
    class_t.clear();
    class_name.clear();
    plane_names.clear();
  }
};

} // namespace tf::arrangement
