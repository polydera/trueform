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

namespace tf {

/// @ingroup core
/// @brief Minimum chunk size for parallel execution.
///
/// Pass to parallel algorithms to bound how finely the range is split:
/// each parallel task processes at least `value` consecutive elements.
struct grain_t {
  std::size_t value;
};

/// @ingroup core
/// @brief Creates a @ref grain_t with the given minimum chunk size.
constexpr auto grain(std::size_t n) -> grain_t { return grain_t{n}; }

} // namespace tf
