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
#include "./closest_metric_point_pair_impl.hpp"

#include <variant>

namespace tf::cpp {

template auto closest_metric_point_pair<float, double, 2>(
    const primitive<float, 2> &, const primitive<double, 2> &)
    -> std::variant<closest_metric_point_pair_result<double>,
                    closest_metric_point_pair_batch_result<double>>;
template auto closest_metric_point_pair<double, float, 2>(
    const primitive<double, 2> &, const primitive<float, 2> &)
    -> std::variant<closest_metric_point_pair_result<double>,
                    closest_metric_point_pair_batch_result<double>>;

} // namespace tf::cpp
