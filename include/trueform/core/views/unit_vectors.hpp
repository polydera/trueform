/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../iter/unit_vector_iterator.hpp"
#include "../range.hpp"

namespace tf::views {
template <std::size_t Dims, typename Range> auto make_unit_vectors(Range &&r) {
  auto begin = tf::iter::make_unit_vector_iterator<Dims>(r.begin());
  auto end = tf::iter::make_unit_vector_iterator<Dims>(r.end());
  return tf::make_range(std::move(begin), std::move(end));
}
} // namespace tf::views
