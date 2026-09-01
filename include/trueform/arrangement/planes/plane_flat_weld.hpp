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

namespace tf::arrangement {

/// One plane-local statement that two input identities reached the same exact
/// projected CDT vertex. The relation is unordered; `plane` is discovery
/// evidence and does not participate in closure.
template <typename Index> struct plane_flat_weld {
  Index flat_0;
  Index flat_1;
  Index plane;
};

} // namespace tf::arrangement
