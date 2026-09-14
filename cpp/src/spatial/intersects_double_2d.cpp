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

/// The carriers this shard tests, each named by its index and arity alone so
/// the matrix can iterate them: a type with a comma in it cannot be a macro
/// argument.
template <typename Index, std::size_t Ngon>
using intersects_mesh_t = mesh<Index, double, 2, Ngon>;
template <typename Index> using intersects_edge_t = edge_mesh<Index, double, 2>;
/// A point cloud names no index of its own.
using intersects_cloud_t = point_cloud<double, 2>;
/// A query, named by its real alone.
template <typename QueryReal>
using intersects_query_t = primitive<QueryReal, 2>;

template auto intersects(const intersects_query_t<double> &,
                         const intersects_query_t<double> &)
    -> intersection_result;

#define TF_CPP_INSTANTIATE_FORM_PRIMITIVE(Form, QueryReal)                     \
  template auto intersects(const Form &,                                       \
                           const intersects_query_t<QueryReal> &)              \
      -> intersection_result;                                                  \
  template auto intersects(const intersects_query_t<QueryReal> &,              \
                           const Form &) -> intersection_result

/// The index and arity axes are the form's, the real axis the query's, so one
/// crossing of the matrix states them all.
#define TF_CPP_INSTANTIATE_MESH_PRIMITIVE(Index, QueryReal, Ngon)              \
  template auto intersects(const intersects_mesh_t<Index, Ngon> &,             \
                           const intersects_query_t<QueryReal> &)              \
      -> intersection_result;                                                  \
  template auto intersects(const intersects_query_t<QueryReal> &,              \
                           const intersects_mesh_t<Index, Ngon> &)             \
      -> intersection_result

#define TF_CPP_INSTANTIATE_EDGE_PRIMITIVE(Index, QueryReal)                    \
  TF_CPP_INSTANTIATE_FORM_PRIMITIVE(intersects_edge_t<Index>, QueryReal)

#define TF_CPP_INSTANTIATE_CLOUD_PRIMITIVE(QueryReal)                          \
  TF_CPP_INSTANTIATE_FORM_PRIMITIVE(intersects_cloud_t, QueryReal)

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_INSTANTIATE_MESH_PRIMITIVE)
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_INSTANTIATE_EDGE_PRIMITIVE)
TF_CPP_MATRIX_FOR_EACH_REAL(TF_CPP_INSTANTIATE_CLOUD_PRIMITIVE)

#undef TF_CPP_INSTANTIATE_CLOUD_PRIMITIVE
#undef TF_CPP_INSTANTIATE_EDGE_PRIMITIVE
#undef TF_CPP_INSTANTIATE_MESH_PRIMITIVE
#undef TF_CPP_INSTANTIATE_FORM_PRIMITIVE

#define TF_CPP_INSTANTIATE_FORM_PAIR(Left, Right)                              \
  template auto intersects(const Left &, const Right &) -> intersection_result

#define TF_CPP_INSTANTIATE_MESH_PAIRS(Index0, Index1, Ngon0, Ngon1)            \
  template auto intersects(const intersects_mesh_t<Index0, Ngon0> &,           \
                           const intersects_mesh_t<Index1, Ngon1> &)           \
      -> intersection_result

#define TF_CPP_INSTANTIATE_MESH_EDGE_PAIRS(Index0, Index1, Ngon)               \
  template auto intersects(const intersects_mesh_t<Index0, Ngon> &,            \
                           const intersects_edge_t<Index1> &)                  \
      -> intersection_result;                                                  \
  template auto intersects(const intersects_edge_t<Index0> &,                  \
                           const intersects_mesh_t<Index1, Ngon> &)            \
      -> intersection_result

#define TF_CPP_INSTANTIATE_EDGE_PAIRS(Index0, Index1)                          \
  TF_CPP_INSTANTIATE_FORM_PAIR(intersects_edge_t<Index0>,                      \
                               intersects_edge_t<Index1>)

#define TF_CPP_INSTANTIATE_MESH_CLOUD_PAIRS(Index, Ngon)                       \
  template auto intersects(const intersects_mesh_t<Index, Ngon> &,             \
                           const intersects_cloud_t &) -> intersection_result; \
  template auto intersects(const intersects_cloud_t &,                         \
                           const intersects_mesh_t<Index, Ngon> &)             \
      -> intersection_result

#define TF_CPP_INSTANTIATE_EDGE_CLOUD_PAIRS(Index)                             \
  TF_CPP_INSTANTIATE_FORM_PAIR(intersects_edge_t<Index>, intersects_cloud_t);  \
  TF_CPP_INSTANTIATE_FORM_PAIR(intersects_cloud_t, intersects_edge_t<Index>)

TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR_NGON_PAIR(TF_CPP_INSTANTIATE_MESH_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR_NGON(TF_CPP_INSTANTIATE_MESH_EDGE_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR(TF_CPP_INSTANTIATE_EDGE_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_NGON(TF_CPP_INSTANTIATE_MESH_CLOUD_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_INSTANTIATE_EDGE_CLOUD_PAIRS)
TF_CPP_INSTANTIATE_FORM_PAIR(intersects_cloud_t, intersects_cloud_t);

#undef TF_CPP_INSTANTIATE_EDGE_CLOUD_PAIRS
#undef TF_CPP_INSTANTIATE_MESH_CLOUD_PAIRS
#undef TF_CPP_INSTANTIATE_EDGE_PAIRS
#undef TF_CPP_INSTANTIATE_MESH_EDGE_PAIRS
#undef TF_CPP_INSTANTIATE_MESH_PAIRS
#undef TF_CPP_INSTANTIATE_FORM_PAIR

} // namespace tf::cpp
