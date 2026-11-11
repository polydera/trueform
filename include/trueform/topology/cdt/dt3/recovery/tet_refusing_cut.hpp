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

/// The cut of a recovery that cuts nothing: every crossing its walk meets
/// stays refused.
struct tet_refusing_cut {
  template <typename Pair, typename Due>
  auto operator()(const Pair &, Due &) const -> bool {
    return false;
  }
};

} // namespace tf::topology::cdt::dt3
