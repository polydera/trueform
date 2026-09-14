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

template auto transformed<float, 2>(const primitive<float, 2> &,
                                    const nd_array<float> &)
    -> primitive<float, 2>;

} // namespace tf::cpp
