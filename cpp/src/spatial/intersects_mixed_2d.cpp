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
#include "./intersects_impl.hpp"

#include <cstddef>

namespace tf::cpp {

/// A mixed shard tests between the two precisions, so it is kept only when the
/// matrix carries both; the index and arity axes are still the matrix's, and a
/// carrier is named by them alone because a type with a comma in it cannot be
/// a macro argument.
template <typename Index, std::size_t Ngon>
using intersects_float_mesh_t = mesh<Index, float, 2, Ngon>;
template <typename Index>
using intersects_float_edge_t = edge_mesh<Index, float, 2>;
template <typename Index, std::size_t Ngon>
using intersects_double_mesh_t = mesh<Index, double, 2, Ngon>;
template <typename Index>
using intersects_double_edge_t = edge_mesh<Index, double, 2>;
/// A point cloud names no index of its own.
using intersects_float_cloud_t = point_cloud<float, 2>;
using intersects_double_cloud_t = point_cloud<double, 2>;
/// A query, named by its real alone.
template <typename QueryReal>
using intersects_query_t = primitive<QueryReal, 2>;

template auto intersects(const intersects_query_t<float> &,
                         const intersects_query_t<double> &)
    -> intersection_result;
template auto intersects(const intersects_query_t<double> &,
                         const intersects_query_t<float> &)
    -> intersection_result;

#define TF_CPP_INSTANTIATE_MIXED_PAIR(FloatForm, DoubleForm)                   \
  template auto intersects(const FloatForm &, const DoubleForm &)              \
      -> intersection_result;                                                  \
  template auto intersects(const DoubleForm &, const FloatForm &)              \
      -> intersection_result

#define TF_CPP_INSTANTIATE_MIXED_MESH_PAIRS(Index0, Index1, Ngon0, Ngon1)      \
  template auto intersects(const intersects_float_mesh_t<Index0, Ngon0> &,     \
                           const intersects_double_mesh_t<Index1, Ngon1> &)    \
      -> intersection_result;                                                  \
  template auto intersects(const intersects_double_mesh_t<Index0, Ngon0> &,    \
                           const intersects_float_mesh_t<Index1, Ngon1> &)     \
      -> intersection_result

#define TF_CPP_INSTANTIATE_MIXED_MESH_EDGE_PAIRS(Index0, Index1, Ngon)         \
  template auto intersects(const intersects_float_mesh_t<Index0, Ngon> &,      \
                           const intersects_double_edge_t<Index1> &)           \
      -> intersection_result;                                                  \
  template auto intersects(const intersects_double_edge_t<Index0> &,           \
                           const intersects_float_mesh_t<Index1, Ngon> &)      \
      -> intersection_result;                                                  \
  template auto intersects(const intersects_float_edge_t<Index0> &,            \
                           const intersects_double_mesh_t<Index1, Ngon> &)     \
      -> intersection_result;                                                  \
  template auto intersects(const intersects_double_mesh_t<Index0, Ngon> &,     \
                           const intersects_float_edge_t<Index1> &)            \
      -> intersection_result

#define TF_CPP_INSTANTIATE_MIXED_EDGE_PAIRS(Index0, Index1)                    \
  TF_CPP_INSTANTIATE_MIXED_PAIR(intersects_float_edge_t<Index0>,               \
                                intersects_double_edge_t<Index1>)

#define TF_CPP_INSTANTIATE_MIXED_MESH_CLOUD_PAIRS(Index, Ngon)                 \
  template auto intersects(const intersects_float_mesh_t<Index, Ngon> &,       \
                           const intersects_double_cloud_t &)                  \
      -> intersection_result;                                                  \
  template auto intersects(const intersects_double_cloud_t &,                  \
                           const intersects_float_mesh_t<Index, Ngon> &)       \
      -> intersection_result;                                                  \
  template auto intersects(const intersects_float_cloud_t &,                   \
                           const intersects_double_mesh_t<Index, Ngon> &)      \
      -> intersection_result;                                                  \
  template auto intersects(const intersects_double_mesh_t<Index, Ngon> &,      \
                           const intersects_float_cloud_t &)                   \
      -> intersection_result

#define TF_CPP_INSTANTIATE_MIXED_EDGE_CLOUD_PAIRS(Index)                       \
  TF_CPP_INSTANTIATE_MIXED_PAIR(intersects_float_edge_t<Index>,                \
                                intersects_double_cloud_t);                    \
  TF_CPP_INSTANTIATE_MIXED_PAIR(intersects_float_cloud_t,                      \
                                intersects_double_edge_t<Index>)

TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR_NGON_PAIR(TF_CPP_INSTANTIATE_MIXED_MESH_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR_NGON(TF_CPP_INSTANTIATE_MIXED_MESH_EDGE_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR(TF_CPP_INSTANTIATE_MIXED_EDGE_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_NGON(TF_CPP_INSTANTIATE_MIXED_MESH_CLOUD_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_INSTANTIATE_MIXED_EDGE_CLOUD_PAIRS)
TF_CPP_INSTANTIATE_MIXED_PAIR(intersects_float_cloud_t,
                              intersects_double_cloud_t);

#undef TF_CPP_INSTANTIATE_MIXED_EDGE_CLOUD_PAIRS
#undef TF_CPP_INSTANTIATE_MIXED_MESH_CLOUD_PAIRS
#undef TF_CPP_INSTANTIATE_MIXED_EDGE_PAIRS
#undef TF_CPP_INSTANTIATE_MIXED_MESH_EDGE_PAIRS
#undef TF_CPP_INSTANTIATE_MIXED_MESH_PAIRS
#undef TF_CPP_INSTANTIATE_MIXED_PAIR

} // namespace tf::cpp
