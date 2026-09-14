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

namespace tf::cpp {

/// @brief Options for one-way or symmetric Chamfer error.
template <typename Real> struct chamfer_error_options {
  Real outlier_proportion = Real{0};
};

} // namespace tf::cpp
