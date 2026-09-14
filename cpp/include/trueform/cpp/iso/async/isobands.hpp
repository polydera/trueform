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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/iso/isobands.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

#define TF_CPP_ISOBANDS_TYPED_ASYNC(Name, Result)                              \
  template <typename Resolver, typename Index, typename Real,                  \
            std::size_t Ngon>                                                  \
  auto Name(Resolver &&resolver, const mesh<Index, Real, 3, Ngon> &value,      \
            const nd_array<Real> &scalars, const nd_array<Real> &cut_values)   \
      -> resolver_result_t<Resolver, Result<Index, Real, Ngon>> {              \
    return submit<Result<Index, Real, Ngon>>(                                  \
        std::forward<Resolver>(resolver), [value, scalars, cut_values] {       \
          return cpp::Name(value, scalars, cut_values);                        \
        });                                                                    \
  }                                                                            \
                                                                               \
  template <typename Index, typename Real, std::size_t Ngon>                   \
  auto Name(const mesh<Index, Real, 3, Ngon> &value,                           \
            const nd_array<Real> &scalars, const nd_array<Real> &cut_values)   \
      -> std::future<Result<Index, Real, Ngon>> {                              \
    return async::Name(future_resolver{}, value, scalars, cut_values);         \
  }

TF_CPP_ISOBANDS_TYPED_ASYNC(isobands, isobands_result)
TF_CPP_ISOBANDS_TYPED_ASYNC(isobands_with_curves, isobands_with_curves_result)

#undef TF_CPP_ISOBANDS_TYPED_ASYNC

#define TF_CPP_ISOBANDS_SELECTED_TYPED_ASYNC(Name, Result)                     \
  template <typename Resolver, typename Index, typename Real,                  \
            std::size_t Ngon>                                                  \
  auto Name(Resolver &&resolver, const mesh<Index, Real, 3, Ngon> &value,      \
            const nd_array<Real> &scalars, const nd_array<Real> &cut_values,   \
            const nd_array<std::int32_t> &selected_bands)                      \
      -> resolver_result_t<Resolver, Result<Index, Real, Ngon>> {              \
    return submit<Result<Index, Real, Ngon>>(                                  \
        std::forward<Resolver>(resolver),                                      \
        [value, scalars, cut_values, selected_bands] {                         \
          return cpp::Name(value, scalars, cut_values, selected_bands);        \
        });                                                                    \
  }                                                                            \
                                                                               \
  template <typename Index, typename Real, std::size_t Ngon>                   \
  auto Name(const mesh<Index, Real, 3, Ngon> &value,                           \
            const nd_array<Real> &scalars, const nd_array<Real> &cut_values,   \
            const nd_array<std::int32_t> &selected_bands)                      \
      -> std::future<Result<Index, Real, Ngon>> {                              \
    return async::Name(future_resolver{}, value, scalars, cut_values,          \
                       selected_bands);                                        \
  }

TF_CPP_ISOBANDS_SELECTED_TYPED_ASYNC(isobands_selected, isobands_result)
TF_CPP_ISOBANDS_SELECTED_TYPED_ASYNC(isobands_with_curves_selected,
                                     isobands_with_curves_result)

#undef TF_CPP_ISOBANDS_SELECTED_TYPED_ASYNC

} // namespace tf::cpp::async
