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

namespace tf {

/// @ingroup core
/// @brief Tag type for explicit zero initialization.
///
/// Used to explicitly request zero-initialization of vectors and points.
/// MSVC fails to zero-initialize members when using value-initialization
/// with inherited constructors (`using Base::Base`), so this tag provides
/// a portable way to guarantee zero-initialized primitives.
struct zero_t {};

/// @ingroup core
/// @brief Tag instance for explicit zero initialization.
inline constexpr zero_t zero;

} // namespace tf
