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
#include "./tet_claim_workspace.hpp"

namespace tf::topology::cdt::dt3 {

template <typename Index>
auto clear_tet_claim_workspace(tet_claim_workspace<Index> &work) -> void {
  work.ownership.clear();
  work.free_next.clear();
  work.tail.clear();
  work.streams.clear();
}

} // namespace tf::topology::cdt::dt3
