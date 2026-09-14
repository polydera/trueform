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
#pragma once

#ifndef TF_CPP_GATHER_REAL
#error "TF_CPP_GATHER_REAL must name the real type for this gather shard"
#endif
#ifndef TF_CPP_GATHER_DIMS
#error "TF_CPP_GATHER_DIMS must name the dimension for this gather shard"
#endif

#include "gather_ids_impl.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::cpp {

/// The carriers this shard gathers between, each named by its index and arity
/// alone so the matrix can iterate them: a type with a comma in it cannot be a
/// macro argument.
template <typename Index, std::size_t Ngon>
using gather_mesh_t = mesh<Index, TF_CPP_GATHER_REAL, TF_CPP_GATHER_DIMS, Ngon>;
template <typename Index>
using gather_edge_t = edge_mesh<Index, TF_CPP_GATHER_REAL, TF_CPP_GATHER_DIMS>;
/// A point cloud names no index of its own: its ids are int32 by construction.
using gather_cloud_t = point_cloud<TF_CPP_GATHER_REAL, TF_CPP_GATHER_DIMS>;

#define TF_CPP_INSTANTIATE_GATHER_FORM_PRIMITIVE(Form, Index)                  \
  template auto gather_ids(                                                    \
      const Form &, const primitive<TF_CPP_GATHER_REAL, TF_CPP_GATHER_DIMS> &) \
      -> nd_array<Index>;                                                      \
  template auto gather_ids_within_distance(                                    \
      const Form &, const primitive<TF_CPP_GATHER_REAL, TF_CPP_GATHER_DIMS> &, \
      TF_CPP_GATHER_REAL) -> nd_array<Index>

#define TF_CPP_INSTANTIATE_GATHER_MESH_PRIMITIVE(Index, Ngon)                  \
  template auto gather_ids(                                                    \
      const gather_mesh_t<Index, Ngon> &,                                      \
      const primitive<TF_CPP_GATHER_REAL, TF_CPP_GATHER_DIMS> &)               \
      -> nd_array<Index>;                                                      \
  template auto gather_ids_within_distance(                                    \
      const gather_mesh_t<Index, Ngon> &,                                      \
      const primitive<TF_CPP_GATHER_REAL, TF_CPP_GATHER_DIMS> &,               \
      TF_CPP_GATHER_REAL) -> nd_array<Index>

#define TF_CPP_INSTANTIATE_GATHER_EDGE_PRIMITIVE(Index)                        \
  TF_CPP_INSTANTIATE_GATHER_FORM_PRIMITIVE(gather_edge_t<Index>, Index)

TF_CPP_MATRIX_FOR_EACH_INDEX_NGON(TF_CPP_INSTANTIATE_GATHER_MESH_PRIMITIVE)
TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_INSTANTIATE_GATHER_EDGE_PRIMITIVE)
TF_CPP_INSTANTIATE_GATHER_FORM_PRIMITIVE(gather_cloud_t, std::int32_t);

#undef TF_CPP_INSTANTIATE_GATHER_EDGE_PRIMITIVE
#undef TF_CPP_INSTANTIATE_GATHER_MESH_PRIMITIVE
#undef TF_CPP_INSTANTIATE_GATHER_FORM_PRIMITIVE

#define TF_CPP_INSTANTIATE_GATHER_PAIR(Left, LeftIndex, Right, RightIndex)     \
  template auto gather_ids(const Left &, const Right &)                        \
      -> nd_array<common_index_t<LeftIndex, RightIndex>>;                      \
  template auto gather_ids_within_distance(const Left &, const Right &,        \
                                           TF_CPP_GATHER_REAL)                 \
      -> nd_array<common_index_t<LeftIndex, RightIndex>>

#define TF_CPP_INSTANTIATE_GATHER_MESH_PAIRS(Index0, Index1, Ngon0, Ngon1)     \
  template auto gather_ids(const gather_mesh_t<Index0, Ngon0> &,               \
                           const gather_mesh_t<Index1, Ngon1> &)               \
      -> nd_array<common_index_t<Index0, Index1>>;                             \
  template auto gather_ids_within_distance(                                    \
      const gather_mesh_t<Index0, Ngon0> &,                                    \
      const gather_mesh_t<Index1, Ngon1> &, TF_CPP_GATHER_REAL)                \
      -> nd_array<common_index_t<Index0, Index1>>

#define TF_CPP_INSTANTIATE_GATHER_MESH_EDGE_PAIRS(Index0, Index1, Ngon)        \
  template auto gather_ids(const gather_mesh_t<Index0, Ngon> &,                \
                           const gather_edge_t<Index1> &)                      \
      -> nd_array<common_index_t<Index0, Index1>>;                             \
  template auto gather_ids_within_distance(                                    \
      const gather_mesh_t<Index0, Ngon> &, const gather_edge_t<Index1> &,      \
      TF_CPP_GATHER_REAL) -> nd_array<common_index_t<Index0, Index1>>;         \
  template auto gather_ids(const gather_edge_t<Index0> &,                      \
                           const gather_mesh_t<Index1, Ngon> &)                \
      -> nd_array<common_index_t<Index0, Index1>>;                             \
  template auto gather_ids_within_distance(                                    \
      const gather_edge_t<Index0> &, const gather_mesh_t<Index1, Ngon> &,      \
      TF_CPP_GATHER_REAL) -> nd_array<common_index_t<Index0, Index1>>

#define TF_CPP_INSTANTIATE_GATHER_EDGE_PAIRS(Index0, Index1)                   \
  TF_CPP_INSTANTIATE_GATHER_PAIR(gather_edge_t<Index0>, Index0,                \
                                 gather_edge_t<Index1>, Index1)

#define TF_CPP_INSTANTIATE_GATHER_MESH_CLOUD_PAIRS(Index, Ngon)                \
  template auto gather_ids(const gather_mesh_t<Index, Ngon> &,                 \
                           const gather_cloud_t &)                             \
      -> nd_array<common_index_t<Index, std::int32_t>>;                        \
  template auto gather_ids_within_distance(                                    \
      const gather_mesh_t<Index, Ngon> &, const gather_cloud_t &,              \
      TF_CPP_GATHER_REAL) -> nd_array<common_index_t<Index, std::int32_t>>;    \
  template auto gather_ids(const gather_cloud_t &,                             \
                           const gather_mesh_t<Index, Ngon> &)                 \
      -> nd_array<common_index_t<std::int32_t, Index>>;                        \
  template auto gather_ids_within_distance(                                    \
      const gather_cloud_t &, const gather_mesh_t<Index, Ngon> &,              \
      TF_CPP_GATHER_REAL) -> nd_array<common_index_t<std::int32_t, Index>>

#define TF_CPP_INSTANTIATE_GATHER_EDGE_CLOUD_PAIRS(Index)                      \
  TF_CPP_INSTANTIATE_GATHER_PAIR(gather_edge_t<Index>, Index, gather_cloud_t,  \
                                 std::int32_t);                                \
  TF_CPP_INSTANTIATE_GATHER_PAIR(gather_cloud_t, std::int32_t,                 \
                                 gather_edge_t<Index>, Index)

TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR_NGON_PAIR(
    TF_CPP_INSTANTIATE_GATHER_MESH_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR_NGON(
    TF_CPP_INSTANTIATE_GATHER_MESH_EDGE_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR(TF_CPP_INSTANTIATE_GATHER_EDGE_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_NGON(TF_CPP_INSTANTIATE_GATHER_MESH_CLOUD_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_INSTANTIATE_GATHER_EDGE_CLOUD_PAIRS)
TF_CPP_INSTANTIATE_GATHER_PAIR(gather_cloud_t, std::int32_t, gather_cloud_t,
                               std::int32_t);

#undef TF_CPP_INSTANTIATE_GATHER_EDGE_CLOUD_PAIRS
#undef TF_CPP_INSTANTIATE_GATHER_MESH_CLOUD_PAIRS
#undef TF_CPP_INSTANTIATE_GATHER_EDGE_PAIRS
#undef TF_CPP_INSTANTIATE_GATHER_MESH_EDGE_PAIRS
#undef TF_CPP_INSTANTIATE_GATHER_MESH_PAIRS
#undef TF_CPP_INSTANTIATE_GATHER_PAIR

} // namespace tf::cpp
