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
#include "./tet_membership.hpp"
#include <type_traits>

namespace tf {

/// @ingroup topology_connectivity
/// @brief Create the tet membership of a tetrahedra range.
///
/// @tparam Index The index type, the tets' own unless stated.
template <typename Index = tf::none_t, typename Policy>
auto make_tet_membership(const tf::tetrahedra<Policy> &tetrahedra) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    return make_tet_membership<
        std::decay_t<decltype(tetrahedra.tetras()[0][0])>>(tetrahedra);
  } else {
    tf::tet_membership<Index> tm;
    tm.build(tetrahedra);
    return tm;
  }
}

} // namespace tf
