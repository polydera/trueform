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

template auto decimated<std::int32_t, float>(
    const mesh<std::int32_t, float, 3> &, float, tf::decimate_config<float>,
    const nd_array<std::int32_t> &) -> remesh_result<std::int32_t, float>;
template auto isotropic_remeshed<std::int32_t, float>(
    const mesh<std::int32_t, float, 3> &, tf::isotropic_remesh_config<float>,
    const nd_array<std::int32_t> &) -> remesh_result<std::int32_t, float>;
template auto simplified<std::int32_t, float>(
    const mesh<std::int32_t, float, 3> &, tf::simplify_config<float>,
    const nd_array<std::int32_t> &) -> remesh_result<std::int32_t, float>;

} // namespace tf::cpp
