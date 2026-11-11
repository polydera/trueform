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
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// The entrance of a cavity's seed, which no facet was entered through.
inline constexpr std::size_t tet_cavity_seed_entrance = 4;

/// The slots a flood asks of a cavity cell, ascending, by the slot the cell
/// was entered through: the three others, and all four of the seed. The
/// cell across the entrance is dead by the flood's own order.
inline constexpr std::uint8_t tet_cavity_sides[5][4] = {
    {1, 2, 3, 0}, {0, 2, 3, 0}, {0, 1, 3, 0}, {0, 1, 2, 0}, {0, 1, 2, 3}};

} // namespace tf::topology::cdt::dt3
