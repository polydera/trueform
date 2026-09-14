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
#include "benchmark_black_box.hpp"

#include <cstddef>

namespace {

const void *volatile escaped_data = nullptr;
volatile std::size_t escaped_size = 0;

#if defined(_MSC_VER)
#define TF_BENCHMARK_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define TF_BENCHMARK_NOINLINE __attribute__((noinline))
#else
#define TF_BENCHMARK_NOINLINE
#endif

} // namespace

extern "C" TF_BENCHMARK_NOINLINE void
tf_cpp_benchmark_black_box(const void *data, std::size_t size) noexcept {
  escaped_data = data;
  escaped_size = size;
}
