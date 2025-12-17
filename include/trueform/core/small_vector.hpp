/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./external/small_vector.hpp"
namespace tf {
template <typename T, unsigned N>
using small_vector = tf::external::llvm_vecsmall::SmallVector<T, N>;
}
