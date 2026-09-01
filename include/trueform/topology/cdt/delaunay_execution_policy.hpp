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

namespace tf::topology::cdt {

/// Execution policies select the carrier schedule, never the geometric
/// operation. Both paths call the same leaf and merge kernels.
struct serial_delaunay_execution_policy {
  static constexpr bool parallel = false;
};

struct parallel_delaunay_execution_policy {
  static constexpr bool parallel = true;
};

} // namespace tf::topology::cdt
