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
#include "transformed_impl.hpp"

namespace tf::cpp {

template auto transformed<float, 3>(const primitive<float, 3> &,
                                    const nd_array<float> &)
    -> primitive<float, 3>;

} // namespace tf::cpp
