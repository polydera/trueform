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

// Implemented in a noinline translation unit compiled without IPO/LTO. The
// opaque call makes the caller's optimizer treat every byte in [data, data +
// size) as observable while the result is alive.
extern "C" void tf_cpp_benchmark_black_box(const void *data,
                                           std::size_t size) noexcept;
