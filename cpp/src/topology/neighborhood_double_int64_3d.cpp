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
#include "neighborhood_impl.hpp"

#include <cstdint>

namespace tf::cpp {

template auto neighborhoods<std::int64_t, double, 3>(
    const offset_blocked_buffer<std::int64_t, std::int64_t> &,
    const nd_array<double> &, double, bool)
    -> offset_blocked_buffer<std::int64_t, std::int64_t>;

} // namespace tf::cpp
