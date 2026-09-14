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

#if !defined(TF_CPP_NEIGHBOR_REAL) || !defined(TF_CPP_NEIGHBOR_DIMS)
#error "TF_CPP_NEIGHBOR_REAL and TF_CPP_NEIGHBOR_DIMS must be defined"
#endif

#include "./neighbor_search_impl.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::cpp {

/// The carriers this shard searches between, each named by its index and arity
/// alone so the matrix can iterate them: a type with a comma in it cannot be a
/// macro argument.
template <typename Index, std::size_t Ngon>
using neighbor_mesh_t =
    mesh<Index, TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS, Ngon>;
template <typename Index>
using neighbor_edge_t =
    edge_mesh<Index, TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS>;
/// A point cloud names no index of its own: its ids are int32 by construction.
using neighbor_cloud_t =
    point_cloud<TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS>;

#define TF_CPP_INSTANTIATE_NEIGHBOR_FORM_PRIMITIVE(Form, Index)                \
  template auto neighbor_search(                                               \
      const Form &,                                                            \
      const primitive<TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS> &,           \
      TF_CPP_NEIGHBOR_REAL)                                                    \
      -> neighbor_result<Index, TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS>;   \
  template auto neighbor_search_batch(                                         \
      const Form &,                                                            \
      const primitive<TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS> &,           \
      TF_CPP_NEIGHBOR_REAL)                                                    \
      -> neighbor_batch_result<Index, TF_CPP_NEIGHBOR_REAL,                    \
                               TF_CPP_NEIGHBOR_DIMS>;                          \
  template auto neighbor_search_knn(                                           \
      const Form &,                                                            \
      const primitive<TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS> &, int,      \
      TF_CPP_NEIGHBOR_REAL)                                                    \
      -> neighbor_knn_result<Index, TF_CPP_NEIGHBOR_REAL,                      \
                             TF_CPP_NEIGHBOR_DIMS>;                            \
  template auto neighbor_search_knn_batch(                                     \
      const Form &,                                                            \
      const primitive<TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS> &, int,      \
      TF_CPP_NEIGHBOR_REAL)                                                    \
      -> neighbor_knn_batch_result<Index, TF_CPP_NEIGHBOR_REAL,                \
                                   TF_CPP_NEIGHBOR_DIMS>

#define TF_CPP_INSTANTIATE_NEIGHBOR_MESH_PRIMITIVE(Index, Ngon)                \
  template auto neighbor_search(                                               \
      const neighbor_mesh_t<Index, Ngon> &,                                    \
      const primitive<TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS> &,           \
      TF_CPP_NEIGHBOR_REAL)                                                    \
      -> neighbor_result<Index, TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS>;   \
  template auto neighbor_search_batch(                                         \
      const neighbor_mesh_t<Index, Ngon> &,                                    \
      const primitive<TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS> &,           \
      TF_CPP_NEIGHBOR_REAL)                                                    \
      -> neighbor_batch_result<Index, TF_CPP_NEIGHBOR_REAL,                    \
                               TF_CPP_NEIGHBOR_DIMS>;                          \
  template auto neighbor_search_knn(                                           \
      const neighbor_mesh_t<Index, Ngon> &,                                    \
      const primitive<TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS> &, int,      \
      TF_CPP_NEIGHBOR_REAL)                                                    \
      -> neighbor_knn_result<Index, TF_CPP_NEIGHBOR_REAL,                      \
                             TF_CPP_NEIGHBOR_DIMS>;                            \
  template auto neighbor_search_knn_batch(                                     \
      const neighbor_mesh_t<Index, Ngon> &,                                    \
      const primitive<TF_CPP_NEIGHBOR_REAL, TF_CPP_NEIGHBOR_DIMS> &, int,      \
      TF_CPP_NEIGHBOR_REAL)                                                    \
      -> neighbor_knn_batch_result<Index, TF_CPP_NEIGHBOR_REAL,                \
                                   TF_CPP_NEIGHBOR_DIMS>

#define TF_CPP_INSTANTIATE_NEIGHBOR_EDGE_PRIMITIVE(Index)                      \
  TF_CPP_INSTANTIATE_NEIGHBOR_FORM_PRIMITIVE(neighbor_edge_t<Index>, Index)

TF_CPP_MATRIX_FOR_EACH_INDEX_NGON(TF_CPP_INSTANTIATE_NEIGHBOR_MESH_PRIMITIVE)
TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_INSTANTIATE_NEIGHBOR_EDGE_PRIMITIVE)
TF_CPP_INSTANTIATE_NEIGHBOR_FORM_PRIMITIVE(neighbor_cloud_t, std::int32_t);

#undef TF_CPP_INSTANTIATE_NEIGHBOR_EDGE_PRIMITIVE
#undef TF_CPP_INSTANTIATE_NEIGHBOR_MESH_PRIMITIVE
#undef TF_CPP_INSTANTIATE_NEIGHBOR_FORM_PRIMITIVE

#define TF_CPP_INSTANTIATE_NEIGHBOR_PAIR(Left, LeftIndex, Right, RightIndex)   \
  template auto neighbor_search(const Left &, const Right &,                   \
                                TF_CPP_NEIGHBOR_REAL)                          \
      -> neighbor_pair_result<LeftIndex, RightIndex, TF_CPP_NEIGHBOR_REAL,     \
                              TF_CPP_NEIGHBOR_DIMS>

#define TF_CPP_INSTANTIATE_NEIGHBOR_MESH_PAIRS(Index0, Index1, Ngon0, Ngon1)   \
  template auto neighbor_search(const neighbor_mesh_t<Index0, Ngon0> &,        \
                                const neighbor_mesh_t<Index1, Ngon1> &,        \
                                TF_CPP_NEIGHBOR_REAL)                          \
      -> neighbor_pair_result<Index0, Index1, TF_CPP_NEIGHBOR_REAL,            \
                              TF_CPP_NEIGHBOR_DIMS>

#define TF_CPP_INSTANTIATE_NEIGHBOR_MESH_EDGE_PAIRS(Index0, Index1, Ngon)      \
  template auto neighbor_search(const neighbor_mesh_t<Index0, Ngon> &,         \
                                const neighbor_edge_t<Index1> &,               \
                                TF_CPP_NEIGHBOR_REAL)                          \
      -> neighbor_pair_result<Index0, Index1, TF_CPP_NEIGHBOR_REAL,            \
                              TF_CPP_NEIGHBOR_DIMS>;                           \
  template auto neighbor_search(const neighbor_edge_t<Index0> &,               \
                                const neighbor_mesh_t<Index1, Ngon> &,         \
                                TF_CPP_NEIGHBOR_REAL)                          \
      -> neighbor_pair_result<Index0, Index1, TF_CPP_NEIGHBOR_REAL,            \
                              TF_CPP_NEIGHBOR_DIMS>

#define TF_CPP_INSTANTIATE_NEIGHBOR_EDGE_PAIRS(Index0, Index1)                 \
  TF_CPP_INSTANTIATE_NEIGHBOR_PAIR(neighbor_edge_t<Index0>, Index0,            \
                                   neighbor_edge_t<Index1>, Index1)

#define TF_CPP_INSTANTIATE_NEIGHBOR_MESH_CLOUD_PAIRS(Index, Ngon)              \
  template auto neighbor_search(const neighbor_mesh_t<Index, Ngon> &,          \
                                const neighbor_cloud_t &,                      \
                                TF_CPP_NEIGHBOR_REAL)                          \
      -> neighbor_pair_result<Index, std::int32_t, TF_CPP_NEIGHBOR_REAL,       \
                              TF_CPP_NEIGHBOR_DIMS>;                           \
  template auto neighbor_search(const neighbor_cloud_t &,                      \
                                const neighbor_mesh_t<Index, Ngon> &,          \
                                TF_CPP_NEIGHBOR_REAL)                          \
      -> neighbor_pair_result<std::int32_t, Index, TF_CPP_NEIGHBOR_REAL,       \
                              TF_CPP_NEIGHBOR_DIMS>

#define TF_CPP_INSTANTIATE_NEIGHBOR_EDGE_CLOUD_PAIRS(Index)                    \
  TF_CPP_INSTANTIATE_NEIGHBOR_PAIR(neighbor_edge_t<Index>, Index,              \
                                   neighbor_cloud_t, std::int32_t);            \
  TF_CPP_INSTANTIATE_NEIGHBOR_PAIR(neighbor_cloud_t, std::int32_t,             \
                                   neighbor_edge_t<Index>, Index)

TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR_NGON_PAIR(
    TF_CPP_INSTANTIATE_NEIGHBOR_MESH_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR_NGON(
    TF_CPP_INSTANTIATE_NEIGHBOR_MESH_EDGE_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR(TF_CPP_INSTANTIATE_NEIGHBOR_EDGE_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX_NGON(TF_CPP_INSTANTIATE_NEIGHBOR_MESH_CLOUD_PAIRS)
TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_INSTANTIATE_NEIGHBOR_EDGE_CLOUD_PAIRS)
TF_CPP_INSTANTIATE_NEIGHBOR_PAIR(neighbor_cloud_t, std::int32_t,
                                 neighbor_cloud_t, std::int32_t);

#undef TF_CPP_INSTANTIATE_NEIGHBOR_EDGE_CLOUD_PAIRS
#undef TF_CPP_INSTANTIATE_NEIGHBOR_MESH_CLOUD_PAIRS
#undef TF_CPP_INSTANTIATE_NEIGHBOR_EDGE_PAIRS
#undef TF_CPP_INSTANTIATE_NEIGHBOR_MESH_EDGE_PAIRS
#undef TF_CPP_INSTANTIATE_NEIGHBOR_MESH_PAIRS
#undef TF_CPP_INSTANTIATE_NEIGHBOR_PAIR

} // namespace tf::cpp
