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

#include <cstdint>
#include <type_traits>

namespace tf::cpp {

/// @brief Default scalar type used to store identity values.
using default_index_t = std::int32_t;

/// @brief Whether T is a supported signed identity-storage scalar.
///
/// This trait describes identity dtype support only. In particular, selecting
/// int64 storage does not extend shape, count, or cardinality APIs beyond their
/// existing INT_MAX limits.
template <typename T>
inline constexpr bool is_supported_index_v =
    std::is_same_v<std::remove_cv_t<T>, std::int32_t> ||
    std::is_same_v<std::remove_cv_t<T>, std::int64_t>;

} // namespace tf::cpp
