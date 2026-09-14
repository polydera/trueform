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

#include "trueform/cpp/reindex/detail/fixed_arity.hpp"

#include <cstddef>

namespace tf::cpp {

/// Fixed tuple-equivalent carrier/result selected by encoded element arity.
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
using reindexed_fixed_carrier_t =
    typename detail::reindexed_fixed_carrier<Index, Real, Dims, Vertices>::type;

template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
using reindexed_fixed_result_t =
    typename detail::reindexed_fixed_result<Index, Real, Dims, Vertices>::type;

} // namespace tf::cpp
