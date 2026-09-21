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
#include "./distance_impl.hpp"

namespace tf::cpp {

template auto distance2(const primitive<float, 2> &,
                        const primitive<double, 2> &)
    -> distance_result<double>;
template auto distance2(const primitive<double, 2> &,
                        const primitive<float, 2> &) -> distance_result<double>;
template auto distance(const primitive<float, 2> &,
                       const primitive<double, 2> &) -> distance_result<double>;
template auto distance(const primitive<double, 2> &,
                       const primitive<float, 2> &) -> distance_result<double>;

template auto distance2(const point_cloud<float, 2> &,
                        const point_cloud<double, 2> &) -> double;
template auto distance2(const point_cloud<double, 2> &,
                        const point_cloud<float, 2> &) -> double;
template auto distance(const point_cloud<float, 2> &,
                       const point_cloud<double, 2> &) -> double;
template auto distance(const point_cloud<double, 2> &,
                       const point_cloud<float, 2> &) -> double;

} // namespace tf::cpp
