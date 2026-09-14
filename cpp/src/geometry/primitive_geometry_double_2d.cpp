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
#include "primitive_geometry_impl.hpp"

namespace tf::cpp {

template auto area<double, 2>(const primitive<double, 2> &)
    -> area_result<double>;
template auto mean_edge_length<double, 2>(const primitive<double, 2> &)
    -> double;

} // namespace tf::cpp
