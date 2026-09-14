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

#include <cstddef>

namespace tf::cpp {

/// @brief The elementwise work — one operation over one element — below which
/// an array kernel runs serially.
///
/// Every array kernel states its own serial cutoff in its own carriers, as
/// this number divided by what one carrier costs. Nothing else decides an
/// array kernel's schedule.
inline constexpr std::size_t parallel_threshold = 150'000;

} // namespace tf::cpp
