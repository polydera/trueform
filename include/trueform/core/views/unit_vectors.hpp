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

#include "../iter/unit_vector_iterator.hpp"
#include "../range.hpp"

namespace tf::views {
template <std::size_t Dims, typename Range> auto make_unit_vectors(Range &&r) {
  auto begin = tf::iter::make_unit_vector_iterator<Dims>(r.begin());
  auto end = tf::iter::make_unit_vector_iterator<Dims>(r.end());
  return tf::make_range(std::move(begin), std::move(end));
}
} // namespace tf::views
