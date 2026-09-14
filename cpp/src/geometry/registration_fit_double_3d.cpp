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
#include "registration_impl.hpp"

namespace tf::cpp {

template auto fit_icp<double, 3>(const point_cloud<double, 3> &,
                                 const point_cloud<double, 3> &,
                                 const fit_icp_options<double> &)
    -> nd_array<double>;
template auto fit_rigid<double, 3>(const point_cloud<double, 3> &,
                                   const point_cloud<double, 3> &)
    -> nd_array<double>;
template auto fit_knn<double, 3>(const point_cloud<double, 3> &,
                                 const point_cloud<double, 3> &,
                                 const fit_knn_options<double> &)
    -> nd_array<double>;
template auto fit_obb<double, 3>(const point_cloud<double, 3> &,
                                 const point_cloud<double, 3> &,
                                 const fit_obb_options &) -> nd_array<double>;
template auto chamfer_error<double, 3>(const point_cloud<double, 3> &,
                                       const point_cloud<double, 3> &,
                                       const chamfer_error_options<double> &)
    -> double;
template auto symmetric_chamfer_error<double, 3>(
    const point_cloud<double, 3> &, const point_cloud<double, 3> &,
    const chamfer_error_options<double> &) -> double;

} // namespace tf::cpp
