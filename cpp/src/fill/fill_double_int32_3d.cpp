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
#include "./fill_impl.hpp"

#include <cstdint>

namespace tf::cpp {

template auto
fill_holes<std::int32_t, double>(const mesh<std::int32_t, double, 3> &,
                                 tf::hole_fill_config)
    -> hole_fill_report<std::int32_t, double>;
template auto fill_holes<std::int32_t, double>(
    const mesh<std::int32_t, double, 3> &, const nd_array<std::int32_t> &,
    tf::hole_fill_config) -> hole_fill_report<std::int32_t, double>;
template auto filled_mesh<std::int32_t, double>(
    const mesh<std::int32_t, double, 3> &,
    const hole_fill_report<std::int32_t, double> &)
    -> tf::polygons_buffer<std::int32_t, double, 3, 3>;

} // namespace tf::cpp
