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

template auto ray_cast<float, float, 2>(const primitive<float, 2> &,
                                        const primitive<float, 2> &,
                                        const ray_cast_options<float> &)
    -> ray_cast_primitive_result<float>;

} // namespace tf::cpp
