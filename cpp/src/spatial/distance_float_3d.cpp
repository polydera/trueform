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

/// A query, named by its real alone.
template <typename QueryReal> using distance_query_t = primitive<QueryReal, 3>;
/// A point cloud names no index of its own.
using point_cloud_type = point_cloud<float, 3>;

#define TF_CPP_INSTANTIATE_DISTANCE(Operation, Left, Right, Result)            \
  template auto Operation(Left, Right) -> Result

TF_CPP_INSTANTIATE_DISTANCE(distance2, const distance_query_t<float> &,
                            const distance_query_t<float> &,
                            distance_result<float>);
TF_CPP_INSTANTIATE_DISTANCE(distance, const distance_query_t<float> &,
                            const distance_query_t<float> &,
                            distance_result<float>);

#define TF_CPP_INSTANTIATE_FP_CLOUD(QueryReal)                                 \
  TF_CPP_INSTANTIATE_DISTANCE(distance2, const point_cloud_type &,             \
                              const distance_query_t<QueryReal> &,             \
                              distance_result<float>);                         \
  TF_CPP_INSTANTIATE_DISTANCE(distance, const point_cloud_type &,              \
                              const distance_query_t<QueryReal> &,             \
                              distance_result<float>)

TF_CPP_MATRIX_FOR_EACH_REAL(TF_CPP_INSTANTIATE_FP_CLOUD)

#define TF_CPP_INSTANTIATE_FF(Operation, Left, Right)                          \
  TF_CPP_INSTANTIATE_DISTANCE(Operation, const Left &, const Right &, float)

TF_CPP_INSTANTIATE_FF(distance2, point_cloud_type, point_cloud_type);
TF_CPP_INSTANTIATE_FF(distance, point_cloud_type, point_cloud_type);

#undef TF_CPP_INSTANTIATE_FF
#undef TF_CPP_INSTANTIATE_FP_CLOUD
#undef TF_CPP_INSTANTIATE_DISTANCE

} // namespace tf::cpp
