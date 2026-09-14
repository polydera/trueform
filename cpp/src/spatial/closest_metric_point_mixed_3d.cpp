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
#include "./closest_metric_point_impl.hpp"

#include <variant>

namespace tf::cpp {

template auto closest_metric_point<float, double, 3>(
    const primitive<float, 3> &, const primitive<double, 3> &)
    -> std::variant<closest_metric_point_result<double>,
                    closest_metric_point_batch_result<double>>;
template auto closest_metric_point<double, float, 3>(
    const primitive<double, 3> &, const primitive<float, 3> &)
    -> std::variant<closest_metric_point_result<double>,
                    closest_metric_point_batch_result<double>>;

} // namespace tf::cpp
