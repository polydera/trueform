/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./sequence_range.hpp"
#include "./zip.hpp"

namespace tf {
template <typename Range> auto enumerate(Range &&r) {
  return tf::zip(tf::make_sequence_range(r.size()), r);
}
} // namespace tf
