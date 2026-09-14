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
#include "./remesh_impl.hpp"

#include <cstdint>

namespace tf::cpp {

template auto decimated<std::int32_t, double>(
    const mesh<std::int32_t, double, 3> &, double, tf::decimate_config<double>,
    const nd_array<std::int32_t> &) -> remesh_result<std::int32_t, double>;
template auto isotropic_remeshed<std::int32_t, double>(
    const mesh<std::int32_t, double, 3> &, tf::isotropic_remesh_config<double>,
    const nd_array<std::int32_t> &) -> remesh_result<std::int32_t, double>;
template auto simplified<std::int32_t, double>(
    const mesh<std::int32_t, double, 3> &, tf::simplify_config<double>,
    const nd_array<std::int32_t> &) -> remesh_result<std::int32_t, double>;

} // namespace tf::cpp
