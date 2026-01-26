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

#include "./external/small_vector.hpp"
namespace tf {

/// @ingroup core_buffers
/// @brief Stack-allocated small vector with heap overflow.
///
/// Stores up to N elements inline; allocates on heap if exceeded.
/// Based on LLVM's SmallVector implementation.
///
/// @tparam T The element type.
/// @tparam N The inline capacity.
template <typename T, unsigned N>
using small_vector = tf::external::llvm_vecsmall::SmallVector<T, N>;
}
