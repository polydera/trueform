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

#include "trueform/arrangement/arrangement_config.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/common_index.hpp"
#include "trueform/cpp/core/detail/concatenated_arity.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/csg/make_boolean.hpp"
#include "trueform/csg/boolean_op.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <utility>
#include <vector>

namespace tf::cpp::async {

#define TF_CPP_DEFINE_ASYNC_BOOLEAN(Operation, Result)                         \
  template <typename Resolver, typename Index0, typename Real,                 \
            typename Index1, std::size_t Ngon0, std::size_t Ngon1>             \
  auto Operation(Resolver &&resolver, const mesh<Index0, Real, 3, Ngon0> &a,   \
                 const mesh<Index1, Real, 3, Ngon1> &b,                        \
                 tf::boolean_op operation,                                     \
                 std::vector<std::int32_t> sheets = {},                        \
                 tf::arrangement_config config = {})                           \
      -> resolver_result_t<                                                    \
          Resolver, Result<common_index_t<Index0, Index1>, Real,               \
                           cpp::detail::concatenated_arity_v<Ngon0, Ngon1>>> { \
    return submit<Result<common_index_t<Index0, Index1>, Real,                 \
                         cpp::detail::concatenated_arity_v<Ngon0, Ngon1>>>(    \
        std::forward<Resolver>(resolver),                                      \
        [a, b, operation, sheets = std::move(sheets), config]() mutable {      \
          return cpp::Operation(a, b, operation, std::move(sheets), config);   \
        });                                                                    \
  }                                                                            \
                                                                               \
  template <typename Index0, typename Real, typename Index1,                   \
            std::size_t Ngon0, std::size_t Ngon1>                              \
  auto Operation(const mesh<Index0, Real, 3, Ngon0> &a,                        \
                 const mesh<Index1, Real, 3, Ngon1> &b,                        \
                 tf::boolean_op operation,                                     \
                 std::vector<std::int32_t> sheets = {},                        \
                 tf::arrangement_config config = {})                           \
      -> std::future<                                                          \
          Result<common_index_t<Index0, Index1>, Real,                         \
                 cpp::detail::concatenated_arity_v<Ngon0, Ngon1>>> {           \
    return async::Operation(future_resolver{}, a, b, operation,                \
                            std::move(sheets), config);                        \
  }

TF_CPP_DEFINE_ASYNC_BOOLEAN(make_boolean, boolean_result)
TF_CPP_DEFINE_ASYNC_BOOLEAN(make_boolean_with_curves,
                            boolean_with_curves_result)

#undef TF_CPP_DEFINE_ASYNC_BOOLEAN

} // namespace tf::cpp::async
