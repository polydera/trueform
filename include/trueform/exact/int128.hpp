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

#if defined(_MSC_VER) && !defined(__clang__)
#include <__msvc_int128.hpp>
#endif

namespace tf::exact {

#if defined(_MSC_VER) && !defined(__clang__)
using int128 = std::_Signed128;
using uint128 = std::_Unsigned128;
#elif defined(__GNUC__) || defined(__clang__)
using int128 = __int128;
using uint128 = unsigned __int128;
#else
#error "No 128-bit integer type available for this compiler"
#endif

} // namespace tf::exact
