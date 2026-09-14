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

/// The cross-index pairs this shard serves exist only when the matrix
/// carries the other index.
#if TF_CPP_MATRIX_HAS_INT32
template <std::size_t Ngon>
using mesh_float_int32_t = mesh<std::int32_t, float, 2, Ngon>;
template <std::size_t Ngon>
using mesh_double_int32_t = mesh<std::int32_t, double, 2, Ngon>;
#endif
template <std::size_t Ngon>
using mesh_float_int64_t = mesh<std::int64_t, float, 2, Ngon>;
template <std::size_t Ngon>
using mesh_double_int64_t = mesh<std::int64_t, double, 2, Ngon>;
using point_cloud_float = point_cloud<float, 2>;
using point_cloud_double = point_cloud<double, 2>;

#define TF_CPP_INSTANTIATE_MIXED_FF(Operation, LeftFloat, LeftDouble,          \
                                    RightFloat, RightDouble)                   \
  template auto Operation(const LeftFloat &, const RightDouble &) -> double;   \
  template auto Operation(const LeftDouble &, const RightFloat &) -> double

#define TF_CPP_INSTANTIATE_MIXED_FF_MESH_MESH(Ngon0, Ngon1)                    \
  TF_CPP_INSTANTIATE_MIXED_FF(                                                 \
      distance2, mesh_float_int64_t<Ngon0>, mesh_double_int64_t<Ngon0>,        \
      mesh_float_int64_t<Ngon1>, mesh_double_int64_t<Ngon1>);                  \
  TF_CPP_INSTANTIATE_MIXED_FF(                                                 \
      distance, mesh_float_int64_t<Ngon0>, mesh_double_int64_t<Ngon0>,         \
      mesh_float_int64_t<Ngon1>, mesh_double_int64_t<Ngon1>)

#define TF_CPP_INSTANTIATE_MIXED_FF_MESH_CLOUD(Ngon)                           \
  TF_CPP_INSTANTIATE_MIXED_FF(distance2, mesh_float_int64_t<Ngon>,             \
                              mesh_double_int64_t<Ngon>, point_cloud_float,    \
                              point_cloud_double);                             \
  TF_CPP_INSTANTIATE_MIXED_FF(distance, mesh_float_int64_t<Ngon>,              \
                              mesh_double_int64_t<Ngon>, point_cloud_float,    \
                              point_cloud_double);                             \
  TF_CPP_INSTANTIATE_MIXED_FF(distance2, point_cloud_float,                    \
                              point_cloud_double, mesh_float_int64_t<Ngon>,    \
                              mesh_double_int64_t<Ngon>);                      \
  TF_CPP_INSTANTIATE_MIXED_FF(distance, point_cloud_float, point_cloud_double, \
                              mesh_float_int64_t<Ngon>,                        \
                              mesh_double_int64_t<Ngon>)

#define TF_CPP_INSTANTIATE_MIXED_FF_CROSS(Ngon0, Ngon1)                        \
  TF_CPP_INSTANTIATE_MIXED_FF(                                                 \
      distance2, mesh_float_int64_t<Ngon0>, mesh_double_int64_t<Ngon0>,        \
      mesh_float_int32_t<Ngon1>, mesh_double_int32_t<Ngon1>);                  \
  TF_CPP_INSTANTIATE_MIXED_FF(                                                 \
      distance, mesh_float_int64_t<Ngon0>, mesh_double_int64_t<Ngon0>,         \
      mesh_float_int32_t<Ngon1>, mesh_double_int32_t<Ngon1>);                  \
  TF_CPP_INSTANTIATE_MIXED_FF(                                                 \
      distance2, mesh_float_int32_t<Ngon0>, mesh_double_int32_t<Ngon0>,        \
      mesh_float_int64_t<Ngon1>, mesh_double_int64_t<Ngon1>);                  \
  TF_CPP_INSTANTIATE_MIXED_FF(                                                 \
      distance, mesh_float_int32_t<Ngon0>, mesh_double_int32_t<Ngon0>,         \
      mesh_float_int64_t<Ngon1>, mesh_double_int64_t<Ngon1>)

TF_CPP_MATRIX_FOR_EACH_NGON_PAIR(TF_CPP_INSTANTIATE_MIXED_FF_MESH_MESH)
TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_MIXED_FF_MESH_CLOUD)
#if TF_CPP_MATRIX_HAS_INT32
TF_CPP_MATRIX_FOR_EACH_NGON_PAIR(TF_CPP_INSTANTIATE_MIXED_FF_CROSS)
#endif

#undef TF_CPP_INSTANTIATE_MIXED_FF_CROSS
#undef TF_CPP_INSTANTIATE_MIXED_FF_MESH_CLOUD
#undef TF_CPP_INSTANTIATE_MIXED_FF_MESH_MESH
#undef TF_CPP_INSTANTIATE_MIXED_FF

} // namespace tf::cpp
