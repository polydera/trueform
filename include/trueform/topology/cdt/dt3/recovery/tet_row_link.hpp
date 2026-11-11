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

namespace tf::topology::cdt::dt3 {

/// A row's place among the rows ticketed at its lower site: the row before
/// it and the row after it there, `-1` past either end
/// (@ref ticket_tet_subsegment).
template <typename Index> struct tet_row_link {
  Index previous;
  Index next;
};

} // namespace tf::topology::cdt::dt3
