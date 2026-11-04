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
#include "./tet_stamps.hpp"

namespace tf::topology::cdt::dt3 {

/// The marks a star gather leaves: one per corner of every cell row, so
/// two gathers of different sites never write one mark and may run
/// together. A mark equal to the generation says the cell is in the star
/// of the site at that corner this generation; a generation gathers each
/// site at most once.
using tet_star_stamps = tet_stamps;

} // namespace tf::topology::cdt::dt3
