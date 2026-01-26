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
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include <algorithm>

namespace tf {

/// @ingroup core_algorithms
/// @brief Copy elements from input to output in parallel.
///
/// This function copies elements from `input` to `output` in parallel using
/// Intel TBB. The operation is divided into blocks and dispatched across
/// threads to improve performance.
///
/// Both `input` and `output` must support random access and have the same size.
/// The `output` range is modified in-place.
///
/// @tparam Range0 The type of the input range. Must provide `.begin()`,
/// `.end()`, and `.size()`.
/// @tparam Range1 The type of the output range. Must support random access and
/// assignment.
/// @param input The source range to copy from.
/// @param output The destination range to copy to.
///
/// @note It is the caller's responsibility to ensure that `output` has at least
/// `input.size()` elements.
template <typename Range0, typename Range1>
auto parallel_copy(const Range0 &input, Range1 &&output) {
  if (input.size() < 1000)
    std::copy(input.begin(), input.end(), output.begin());
  else
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, input.size()),
        [&input, &output](const tbb::blocked_range<std::size_t> &range) {
          std::copy(input.begin() + range.begin(), input.begin() + range.end(),
                    output.begin() + range.begin());
        });
}

} // namespace tf
