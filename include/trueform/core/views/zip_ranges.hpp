/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../iter/zip_iterator.hpp"
#include "../range.hpp"

namespace tf {
template <typename Range> auto zip_ranges(Range &&r) {
  return tf::make_range(r);
}
template <typename... Ranges> auto zip_ranges(Ranges &&...r) {
  return tf::make_range(iter::make_zip_iterator(r.begin()...),
                        iter::make_zip_iterator(r.end()...));
}
} // namespace tf
