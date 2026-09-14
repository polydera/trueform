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

template auto area<float, 3>(const primitive<float, 3> &) -> area_result<float>;
template auto mean_edge_length<float, 3>(const primitive<float, 3> &) -> float;
template auto normals<float>(const primitive<float, 3> &) -> nd_array<float>;

} // namespace tf::cpp
