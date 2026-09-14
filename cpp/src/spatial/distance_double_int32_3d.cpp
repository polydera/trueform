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

#include <cstddef>
#include <cstdint>

namespace tf::cpp {

/// A query, named by its real alone.
template <typename QueryReal>
using distance_query_t = primitive<QueryReal, 3>;
template <std::size_t Ngon> using mesh_t = mesh<std::int32_t, double, 3, Ngon>;
using point_cloud_type = point_cloud<double, 3>;

#define TF_CPP_INSTANTIATE_DISTANCE(Operation, Left, Right, Result)            \
  template auto Operation(Left, Right) -> Result

TF_CPP_INSTANTIATE_DISTANCE(distance2, const distance_query_t<double> &,
                            const distance_query_t<double> &,
                            distance_result<double>);
TF_CPP_INSTANTIATE_DISTANCE(distance, const distance_query_t<double> &,
                            const distance_query_t<double> &,
                            distance_result<double>);

#define TF_CPP_INSTANTIATE_FP(Operation, Form, QueryReal)                      \
  TF_CPP_INSTANTIATE_DISTANCE(Operation, const Form &,                         \
                              const distance_query_t<QueryReal> &,             \
                              distance_result<double>)

#define TF_CPP_INSTANTIATE_FP_MESH(QueryReal, Ngon)                            \
  TF_CPP_INSTANTIATE_FP(distance2, mesh_t<Ngon>, QueryReal);                   \
  TF_CPP_INSTANTIATE_FP(distance, mesh_t<Ngon>, QueryReal)

#define TF_CPP_INSTANTIATE_FP_CLOUD(QueryReal)                                 \
  TF_CPP_INSTANTIATE_FP(distance2, point_cloud_type, QueryReal);               \
  TF_CPP_INSTANTIATE_FP(distance, point_cloud_type, QueryReal)

TF_CPP_MATRIX_FOR_EACH_REAL_NGON(TF_CPP_INSTANTIATE_FP_MESH)
TF_CPP_MATRIX_FOR_EACH_REAL(TF_CPP_INSTANTIATE_FP_CLOUD)

#define TF_CPP_INSTANTIATE_FF(Operation, Left, Right)                          \
  TF_CPP_INSTANTIATE_DISTANCE(Operation, const Left &, const Right &, double)

#define TF_CPP_INSTANTIATE_FF_MESH_MESH(Ngon0, Ngon1)                          \
  TF_CPP_INSTANTIATE_FF(distance2, mesh_t<Ngon0>, mesh_t<Ngon1>);              \
  TF_CPP_INSTANTIATE_FF(distance, mesh_t<Ngon0>, mesh_t<Ngon1>)

#define TF_CPP_INSTANTIATE_FF_MESH_CLOUD(Ngon)                                 \
  TF_CPP_INSTANTIATE_FF(distance2, mesh_t<Ngon>, point_cloud_type);            \
  TF_CPP_INSTANTIATE_FF(distance, mesh_t<Ngon>, point_cloud_type);             \
  TF_CPP_INSTANTIATE_FF(distance2, point_cloud_type, mesh_t<Ngon>);            \
  TF_CPP_INSTANTIATE_FF(distance, point_cloud_type, mesh_t<Ngon>)

TF_CPP_MATRIX_FOR_EACH_NGON_PAIR(TF_CPP_INSTANTIATE_FF_MESH_MESH)
TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_FF_MESH_CLOUD)
TF_CPP_INSTANTIATE_FF(distance2, point_cloud_type, point_cloud_type);
TF_CPP_INSTANTIATE_FF(distance, point_cloud_type, point_cloud_type);

#undef TF_CPP_INSTANTIATE_FF_MESH_CLOUD
#undef TF_CPP_INSTANTIATE_FF_MESH_MESH
#undef TF_CPP_INSTANTIATE_FF
#undef TF_CPP_INSTANTIATE_FP_CLOUD
#undef TF_CPP_INSTANTIATE_FP_MESH
#undef TF_CPP_INSTANTIATE_FP
#undef TF_CPP_INSTANTIATE_DISTANCE

} // namespace tf::cpp
