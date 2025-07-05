/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include <algorithm>

namespace tf {

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

} // namespace tf
