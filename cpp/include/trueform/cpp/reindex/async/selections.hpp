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

#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/reindex/by_ids.hpp"
#include "trueform/cpp/reindex/by_ids_on_points.hpp"
#include "trueform/cpp/reindex/by_mask.hpp"
#include "trueform/cpp/reindex/by_mask_on_points.hpp"
#include "trueform/cpp/reindex/selection_result.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Index, typename Real, std::size_t Dims, typename Resolver>
auto reindexed_by_ids(Resolver &&resolver, const nd_array<Real> &points,
                      const nd_array<Index> &ids) {
  return submit<nd_array<Real>>(
      std::forward<Resolver>(resolver), [points, ids] {
        return cpp::reindexed_by_ids<Index, Real, Dims>(points, ids);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids(const nd_array<Real> &points, const nd_array<Index> &ids)
    -> std::future<nd_array<Real>> {
  return async::reindexed_by_ids<Index, Real, Dims>(future_resolver{}, points,
                                                    ids);
}

template <typename Index, typename Real, std::size_t Dims, typename Resolver>
auto reindexed_by_ids_with_maps(Resolver &&resolver,
                                const nd_array<Real> &points,
                                const nd_array<Index> &ids) {
  return submit<reindexed_points_result<Index, Real>>(
      std::forward<Resolver>(resolver), [points, ids] {
        return cpp::reindexed_by_ids_with_maps<Index, Real, Dims>(points, ids);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids_with_maps(const nd_array<Real> &points,
                                const nd_array<Index> &ids)
    -> std::future<reindexed_points_result<Index, Real>> {
  return async::reindexed_by_ids_with_maps<Index, Real, Dims>(future_resolver{},
                                                              points, ids);
}

template <typename Index, typename Real, std::size_t Dims, typename Resolver>
auto reindexed_by_mask(Resolver &&resolver, const nd_array<Real> &points,
                       const nd_array<std::int8_t> &mask) {
  return submit<nd_array<Real>>(
      std::forward<Resolver>(resolver), [points, mask] {
        return cpp::reindexed_by_mask<Index, Real, Dims>(points, mask);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_mask(const nd_array<Real> &points,
                       const nd_array<std::int8_t> &mask)
    -> std::future<nd_array<Real>> {
  return async::reindexed_by_mask<Index, Real, Dims>(future_resolver{}, points,
                                                     mask);
}

template <typename Index, typename Real, std::size_t Dims, typename Resolver>
auto reindexed_by_mask_with_maps(Resolver &&resolver,
                                 const nd_array<Real> &points,
                                 const nd_array<std::int8_t> &mask) {
  return submit<reindexed_points_result<Index, Real>>(
      std::forward<Resolver>(resolver), [points, mask] {
        return cpp::reindexed_by_mask_with_maps<Index, Real, Dims>(points,
                                                                   mask);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_mask_with_maps(const nd_array<Real> &points,
                                 const nd_array<std::int8_t> &mask)
    -> std::future<reindexed_points_result<Index, Real>> {
  return async::reindexed_by_mask_with_maps<Index, Real, Dims>(
      future_resolver{}, points, mask);
}

// A raw array is copied before resolver invocation, so resolver-side
// replacement cannot change the work the job does.
template <typename Index, typename Real, std::size_t Dims, typename Resolver,
          typename Connectivity, typename Selection, typename Function>
auto submit_array_selection(Resolver &&resolver, const Connectivity &faces,
                            const nd_array<Real> &points,
                            const Selection &selection, Function function) {
  using result_type = decltype(function(faces, points, selection));
  return submit<result_type>(std::forward<Resolver>(resolver),
                             [faces, points, selection, function] {
                               return function(faces, points, selection);
                             });
}

#define TF_CPP_TYPED_ASYNC_ARRAY_SELECTOR(Name, SelectionType)                 \
  template <typename Index, typename Real, std::size_t Dims,                   \
            typename Resolver>                                                 \
  auto Name(Resolver &&resolver, const nd_array<Index> &faces,                 \
            const nd_array<Real> &points,                                      \
            const nd_array<SelectionType> &selection) {                        \
    return submit_array_selection<Index, Real, Dims>(                          \
        std::forward<Resolver>(resolver), faces, points, selection,            \
        [](const auto &owned_faces, const auto &owned_points,                  \
           const auto &owned_selection) {                                      \
          return cpp::Name<Index, Real, Dims>(owned_faces, owned_points,       \
                                              owned_selection);                \
        });                                                                    \
  }                                                                            \
  template <typename Index, typename Real, std::size_t Dims>                   \
  auto Name(const nd_array<Index> &faces, const nd_array<Real> &points,        \
            const nd_array<SelectionType> &selection) {                        \
    return async::Name<Index, Real, Dims>(future_resolver{}, faces, points,    \
                                          selection);                          \
  }                                                                            \
  template <typename Index, typename Real, std::size_t Dims,                   \
            typename Resolver>                                                 \
  auto Name(Resolver &&resolver,                                               \
            const offset_blocked_buffer<Index, Index> &faces,                  \
            const nd_array<Real> &points,                                      \
            const nd_array<SelectionType> &selection) {                        \
    return submit_array_selection<Index, Real, Dims>(                          \
        std::forward<Resolver>(resolver), faces, points, selection,            \
        [](const auto &owned_faces, const auto &owned_points,                  \
           const auto &owned_selection) {                                      \
          return cpp::Name<Index, Real, Dims>(owned_faces, owned_points,       \
                                              owned_selection);                \
        });                                                                    \
  }                                                                            \
  template <typename Index, typename Real, std::size_t Dims>                   \
  auto Name(const offset_blocked_buffer<Index, Index> &faces,                  \
            const nd_array<Real> &points,                                      \
            const nd_array<SelectionType> &selection) {                        \
    return async::Name<Index, Real, Dims>(future_resolver{}, faces, points,    \
                                          selection);                          \
  }

TF_CPP_TYPED_ASYNC_ARRAY_SELECTOR(reindexed_by_ids, Index)
TF_CPP_TYPED_ASYNC_ARRAY_SELECTOR(reindexed_by_ids_with_maps, Index)
TF_CPP_TYPED_ASYNC_ARRAY_SELECTOR(reindexed_by_mask, std::int8_t)
TF_CPP_TYPED_ASYNC_ARRAY_SELECTOR(reindexed_by_mask_with_maps, std::int8_t)
TF_CPP_TYPED_ASYNC_ARRAY_SELECTOR(reindexed_by_ids_on_points, Index)
TF_CPP_TYPED_ASYNC_ARRAY_SELECTOR(reindexed_by_mask_on_points, std::int8_t)

#undef TF_CPP_TYPED_ASYNC_ARRAY_SELECTOR

#define TF_CPP_TYPED_ASYNC_FIXED_ARITY_SELECTOR(Name, SelectionType)           \
  template <typename Index, typename Real, std::size_t Dims,                   \
            std::size_t Vertices, typename Resolver,                           \
            std::enable_if_t<Vertices == 2 || Vertices == 3, int> = 0>         \
  auto Name(Resolver &&resolver, const nd_array<Index> &elements,              \
            const nd_array<Real> &points,                                      \
            const nd_array<SelectionType> &selection) {                        \
    return submit_array_selection<Index, Real, Dims>(                          \
        std::forward<Resolver>(resolver), elements, points, selection,         \
        [](const auto &owned_elements, const auto &owned_points,               \
           const auto &owned_selection) {                                      \
          return cpp::Name<Index, Real, Dims, Vertices>(                       \
              owned_elements, owned_points, owned_selection);                  \
        });                                                                    \
  }                                                                            \
  template <typename Index, typename Real, std::size_t Dims,                   \
            std::size_t Vertices,                                              \
            std::enable_if_t<Vertices == 2 || Vertices == 3, int> = 0>         \
  auto Name(const nd_array<Index> &elements, const nd_array<Real> &points,     \
            const nd_array<SelectionType> &selection) {                        \
    return async::Name<Index, Real, Dims, Vertices>(                           \
        future_resolver{}, elements, points, selection);                       \
  }

TF_CPP_TYPED_ASYNC_FIXED_ARITY_SELECTOR(reindexed_by_ids, Index)
TF_CPP_TYPED_ASYNC_FIXED_ARITY_SELECTOR(reindexed_by_ids_with_maps, Index)
TF_CPP_TYPED_ASYNC_FIXED_ARITY_SELECTOR(reindexed_by_mask, std::int8_t)
TF_CPP_TYPED_ASYNC_FIXED_ARITY_SELECTOR(reindexed_by_mask_with_maps,
                                        std::int8_t)
TF_CPP_TYPED_ASYNC_FIXED_ARITY_SELECTOR(reindexed_by_ids_on_points, Index)
TF_CPP_TYPED_ASYNC_FIXED_ARITY_SELECTOR(reindexed_by_mask_on_points,
                                        std::int8_t)

#undef TF_CPP_TYPED_ASYNC_FIXED_ARITY_SELECTOR

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon, typename Selection, typename Function>
auto submit_mesh_selection(Resolver &&resolver,
                           const mesh<Index, Real, Dims, Ngon> &value,
                           const Selection &selection, Function function) {
  using result_type = decltype(function(value, selection));
  return submit<result_type>(
      std::forward<Resolver>(resolver),
      [value, selection, function] { return function(value, selection); });
}

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          typename Selection, typename Function>
auto submit_edge_selection(Resolver &&resolver,
                           const edge_mesh<Index, Real, Dims> &value,
                           const Selection &selection, Function function) {
  using result_type = decltype(function(value, selection));
  return submit<result_type>(
      std::forward<Resolver>(resolver),
      [value, selection, function] { return function(value, selection); });
}

#define TF_CPP_TYPED_ASYNC_MESH_SELECTOR(Name, SelectionType)                  \
  template <typename Resolver, typename Index, typename Real,                  \
            std::size_t Dims, std::size_t Ngon>                                \
  auto Name(Resolver &&resolver, const mesh<Index, Real, Dims, Ngon> &value,   \
            const nd_array<SelectionType> &selection) {                        \
    return submit_mesh_selection(                                              \
        std::forward<Resolver>(resolver), value, selection,                    \
        [](const auto &owned, const auto &owned_selection) {                   \
          return cpp::Name<Index, Real, Dims, Ngon>(owned, owned_selection);   \
        });                                                                    \
  }                                                                            \
  template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon> \
  auto Name(const mesh<Index, Real, Dims, Ngon> &value,                        \
            const nd_array<SelectionType> &selection) {                        \
    return async::Name<future_resolver, Index, Real, Dims, Ngon>(              \
        future_resolver{}, value, selection);                                  \
  }

TF_CPP_TYPED_ASYNC_MESH_SELECTOR(reindexed_by_ids, Index)
TF_CPP_TYPED_ASYNC_MESH_SELECTOR(reindexed_by_ids_with_maps, Index)
TF_CPP_TYPED_ASYNC_MESH_SELECTOR(reindexed_by_mask, std::int8_t)
TF_CPP_TYPED_ASYNC_MESH_SELECTOR(reindexed_by_mask_with_maps, std::int8_t)
TF_CPP_TYPED_ASYNC_MESH_SELECTOR(reindexed_by_ids_on_points, Index)
TF_CPP_TYPED_ASYNC_MESH_SELECTOR(reindexed_by_ids_on_points_with_maps, Index)
TF_CPP_TYPED_ASYNC_MESH_SELECTOR(reindexed_by_mask_on_points, std::int8_t)
TF_CPP_TYPED_ASYNC_MESH_SELECTOR(reindexed_by_mask_on_points_with_maps,
                                 std::int8_t)

#undef TF_CPP_TYPED_ASYNC_MESH_SELECTOR

#define TF_CPP_TYPED_ASYNC_EDGE_SELECTOR(Name, SelectionType)                  \
  template <typename Resolver, typename Index, typename Real,                  \
            std::size_t Dims>                                                  \
  auto Name(Resolver &&resolver, const edge_mesh<Index, Real, Dims> &value,    \
            const nd_array<SelectionType> &selection) {                        \
    return submit_edge_selection(                                              \
        std::forward<Resolver>(resolver), value, selection,                    \
        [](const auto &owned, const auto &owned_selection) {                   \
          return cpp::Name<Index, Real, Dims>(owned, owned_selection);         \
        });                                                                    \
  }                                                                            \
  template <typename Index, typename Real, std::size_t Dims>                   \
  auto Name(const edge_mesh<Index, Real, Dims> &value,                         \
            const nd_array<SelectionType> &selection) {                        \
    return async::Name<future_resolver, Index, Real, Dims>(future_resolver{},  \
                                                           value, selection);  \
  }

TF_CPP_TYPED_ASYNC_EDGE_SELECTOR(reindexed_by_ids, Index)
TF_CPP_TYPED_ASYNC_EDGE_SELECTOR(reindexed_by_ids_with_maps, Index)
TF_CPP_TYPED_ASYNC_EDGE_SELECTOR(reindexed_by_mask, std::int8_t)
TF_CPP_TYPED_ASYNC_EDGE_SELECTOR(reindexed_by_mask_with_maps, std::int8_t)
TF_CPP_TYPED_ASYNC_EDGE_SELECTOR(reindexed_by_ids_on_points, Index)
TF_CPP_TYPED_ASYNC_EDGE_SELECTOR(reindexed_by_ids_on_points_with_maps, Index)
TF_CPP_TYPED_ASYNC_EDGE_SELECTOR(reindexed_by_mask_on_points, std::int8_t)
TF_CPP_TYPED_ASYNC_EDGE_SELECTOR(reindexed_by_mask_on_points_with_maps,
                                 std::int8_t)

#undef TF_CPP_TYPED_ASYNC_EDGE_SELECTOR

} // namespace tf::cpp::async
