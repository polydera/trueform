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

template auto fit_icp<float, 2>(const point_cloud<float, 2> &,
                                const point_cloud<float, 2> &,
                                const fit_icp_options<float> &)
    -> nd_array<float>;
template auto fit_rigid<float, 2>(const point_cloud<float, 2> &,
                                  const point_cloud<float, 2> &)
    -> nd_array<float>;
template auto fit_knn<float, 2>(const point_cloud<float, 2> &,
                                const point_cloud<float, 2> &,
                                const fit_knn_options<float> &)
    -> nd_array<float>;
template auto fit_obb<float, 2>(const point_cloud<float, 2> &,
                                const point_cloud<float, 2> &,
                                const fit_obb_options &) -> nd_array<float>;
template auto chamfer_error<float, 2>(const point_cloud<float, 2> &,
                                      const point_cloud<float, 2> &,
                                      const chamfer_error_options<float> &)
    -> float;
template auto symmetric_chamfer_error<float, 2>(
    const point_cloud<float, 2> &, const point_cloud<float, 2> &,
    const chamfer_error_options<float> &) -> float;

} // namespace tf::cpp
