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
#pragma once

#include "trueform/cpp/core/index_type.hpp"

#include <type_traits>

namespace tf::cpp::detail {

/// The (index, real) pairs an outer shell has a compiled kernel for. One
/// producer of that fact, read by the entry and by its async front.
template <typename Index, typename Real>
inline constexpr bool is_supported_outer_shell_v =
    (std::is_same_v<Real, float> || std::is_same_v<Real, double>) &&
    is_supported_index_v<Index> &&
    std::is_same_v<Index, std::remove_cv_t<Index>>;

} // namespace tf::cpp::detail
