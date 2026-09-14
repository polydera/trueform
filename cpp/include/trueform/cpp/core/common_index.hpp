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

#include "trueform/cpp/core/detail/common_index.hpp"

namespace tf::cpp {

/// @brief Common signed identity-storage type for supported index dtypes.
template <typename... IndexTs>
using common_index_t = typename detail::common_index<IndexTs...>::type;

} // namespace tf::cpp
