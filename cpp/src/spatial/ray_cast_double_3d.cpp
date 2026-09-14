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
#include "./ray_cast_impl.hpp"

namespace tf::cpp {

template auto ray_cast(const primitive<double> &, const primitive<double> &,
                       const ray_cast_options<double> &)
    -> ray_cast_primitive_result<double>;

} // namespace tf::cpp
