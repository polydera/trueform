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
#include "../../core/none.hpp"
#include "../../exact/int32.hpp"
#include "../../exact/int64.hpp"
#include "./tet_entrance_budget.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tf::tetmesh {

/// The instantiations a tet mesh may run on, cheapest first: the `Int`
/// lattice and the `Coord` a site is stored in. Rung 0 is int32 stored in
/// int64, rung 1 int64 in int64; a site below the frame is read on the
/// rung's deep lattice (@ref tf::topology::cdt::dt3::tet_deep_lattice).
template <std::size_t Rung> struct tet_entrance_rung_types;
template <> struct tet_entrance_rung_types<0> {
  using int_type = tf::exact::int32;
  using coord_type = std::int64_t;
};
template <> struct tet_entrance_rung_types<1> {
  using int_type = tf::exact::int64;
  using coord_type = std::int64_t;
};

/// Whether `Rung` may run a surface of `RealT` coordinates: its lattice
/// (@ref tet_entrance_budget) holds the input's precision and is the one an
/// `IntRequest` states.
template <std::size_t Rung, typename IntRequest, typename RealT>
inline constexpr bool tet_entrance_rung_admits =
    std::numeric_limits<RealT>::digits <=
        tet_entrance_budget<typename tet_entrance_rung_types<Rung>::int_type,
                            typename tet_entrance_rung_types<Rung>::coord_type>::
            lattice_bits &&
    (std::is_same_v<IntRequest, tf::none_t> ||
     std::is_same_v<IntRequest,
                    typename tet_entrance_rung_types<Rung>::int_type>);

} // namespace tf::tetmesh
