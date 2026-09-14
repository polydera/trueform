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

#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>

namespace tf::cpp {

auto logical_not(const nd_array<std::int8_t> &a) -> nd_array<std::int8_t>;
auto logical_not_inplace(nd_array<std::int8_t> &a) -> void;
auto logical_and(const nd_array<std::int8_t> &a, const nd_array<std::int8_t> &b)
    -> nd_array<std::int8_t>;
auto logical_or(const nd_array<std::int8_t> &a, const nd_array<std::int8_t> &b)
    -> nd_array<std::int8_t>;

} // namespace tf::cpp
