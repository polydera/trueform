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

#include "trueform/core/small_vector.hpp"

/// The shape arithmetic an elementwise kernel walks a broadcast with. It is
/// this tier's own machinery, not a caller's vocabulary.
namespace tf::cpp::detail {

auto broadcast_shape(const tf::small_vector<int, 3> &a,
                     const tf::small_vector<int, 3> &b)
    -> tf::small_vector<int, 3>;
auto compute_strides(const tf::small_vector<int, 3> &shape)
    -> tf::small_vector<int, 3>;
auto broadcast_strides(const tf::small_vector<int, 3> &source_shape,
                       const tf::small_vector<int, 3> &output_shape)
    -> tf::small_vector<int, 3>;
auto broadcast_index(int flat_index,
                     const tf::small_vector<int, 3> &output_strides,
                     const tf::small_vector<int, 3> &broadcasted_strides,
                     int dimensions) -> int;
auto shapes_equal(const tf::small_vector<int, 3> &a,
                  const tf::small_vector<int, 3> &b) -> bool;
auto total_size(const tf::small_vector<int, 3> &shape) -> int;

} // namespace tf::cpp::detail
