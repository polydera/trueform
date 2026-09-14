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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/index_map.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/reindex/mesh.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto reindexed(Resolver &&resolver, const mesh<Index, Real, Dims, Ngon> &value,
               const index_map<Index> &face_map,
               const index_map<Index> &point_map) {
  return submit<tf::polygons_buffer<Index, Real, Dims, Ngon>>(
      std::forward<Resolver>(resolver), [value, face_map, point_map] {
        return cpp::reindexed<Index, Real, Dims, Ngon>(value, face_map,
                                                       point_map);
      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed(const mesh<Index, Real, Dims, Ngon> &value,
               const index_map<Index> &face_map,
               const index_map<Index> &point_map)
    -> std::future<tf::polygons_buffer<Index, Real, Dims, Ngon>> {
  return async::reindexed<future_resolver, Index, Real, Dims, Ngon>(
      future_resolver{}, value, face_map, point_map);
}

} // namespace tf::cpp::async
