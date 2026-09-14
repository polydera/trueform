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
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/spatial/distance.hpp"
#include "trueform/cpp/spatial/primitive.hpp"
#include "trueform/cpp/spatial/signed_distance.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename FormReal,
          std::size_t Dims, std::size_t Ngon, typename PrimitiveReal,
          std::enable_if_t<Dims == 3, int> = 0>
auto signed_distance(Resolver &&resolver,
                     const mesh<Index, FormReal, Dims, Ngon> &form,
                     const primitive<PrimitiveReal, Dims> &query) {
  using result_type = distance_result<FormReal>;
  return submit<result_type>(std::forward<Resolver>(resolver), [form, query] {
    return cpp::signed_distance(form, query);
  });
}

template <typename Index, typename FormReal, std::size_t Dims, std::size_t Ngon,
          typename PrimitiveReal, std::enable_if_t<Dims == 3, int> = 0>
auto signed_distance(const mesh<Index, FormReal, Dims, Ngon> &form,
                     const primitive<PrimitiveReal, Dims> &query)
    -> std::future<distance_result<FormReal>> {
  return async::signed_distance(future_resolver{}, form, query);
}

} // namespace tf::cpp::async
