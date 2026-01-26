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
#include "../checked.hpp"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include <algorithm>

namespace tf {

/// @ingroup core_algorithms
/// @brief Transform elements from input to output in parallel.
///
/// Applies a transformation function to each element of the input range
/// and stores the result in the output range.
///
/// @tparam Range0 The input range type.
/// @tparam Range1 The output range type (must be pre-allocated).
/// @tparam F A unary function: `output_element(input_element)`.
/// @param input The source range.
/// @param output The destination range (must have same size as input).
/// @param transform The transformation function.
template <typename Range0, typename Range1, typename F>
auto parallel_transform(const Range0 &input, Range1 &&output,
                        const F &transform) {
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, input.size()),
                    [&input, &output,
                     &transform](const tbb::blocked_range<std::size_t> &range) {
                      std::transform(input.begin() + range.begin(),
                                     input.begin() + range.end(),
                                     output.begin() + range.begin(), transform);
                    });
}

/// @ingroup core_algorithms
/// @brief Transform elements with checked execution.
///
/// Falls back to sequential execution for ranges smaller than 1000 elements.
template <typename Range0, typename Range1, typename F>
auto parallel_transform(const Range0 &input, Range1 &&output,
                        const F &transform, tf::checked_t) {
  if (input.size() < 1000) {
    std::transform(input.begin(), input.end(), output.begin(), transform);
  } else {
    parallel_transform(input, output, transform);
  }
}

} // namespace tf
