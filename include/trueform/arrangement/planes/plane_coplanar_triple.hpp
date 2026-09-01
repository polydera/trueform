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

/// @brief A coincident stack at triangle grain, in the sorted-by-survivor
///        form the inclusion and reverse-side consumers binary-search.
template <typename Index> struct plane_coplanar_triple {
  Index survivor, dead;
  char opposing;
};

} // namespace tf::arrangement
