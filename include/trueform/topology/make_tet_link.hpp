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
#include "../core/none.hpp"
#include "../core/tetrahedra.hpp"
#include "./policy/tet_membership.hpp"
#include "./tet_link.hpp"
#include "./tet_membership.hpp"
#include <type_traits>

namespace tf {

/// @ingroup topology_connectivity
/// @brief Create the tet link of a tetrahedra range, over its tagged tet
/// membership or one built for the call.
///
/// @tparam Index The index type, the tets' own unless stated.
template <typename Index = tf::none_t, typename Policy>
auto make_tet_link(const tf::tetrahedra<Policy> &tetrahedra) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    return make_tet_link<std::decay_t<decltype(tetrahedra.tetras()[0][0])>>(
        tetrahedra);
  } else {
    if constexpr (tf::has_tet_membership_policy<Policy>) {
      tf::tet_link<Index> tl;
      tl.build(tetrahedra.tetras(), tetrahedra.tet_membership());
      return tl;
    } else {
      tf::tet_membership<Index> tm;
      tm.build(tetrahedra);
      tf::tet_link<Index> tl;
      tl.build(tetrahedra.tetras(), tm);
      return tl;
    }
  }
}

} // namespace tf
