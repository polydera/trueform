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

#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>

namespace tf::cpp {

template <typename T> auto assign_scalar(nd_array<T> &target, T scalar) -> void;
template <typename T>
auto assign_array(nd_array<T> &target, const nd_array<T> &source) -> void;
template <typename T>
auto assign_indexed_scalar(nd_array<T> &target,
                           const nd_array<std::int32_t> &indices, T scalar)
    -> void;
template <typename T>
auto assign_indexed_array(nd_array<T> &target,
                          const nd_array<std::int32_t> &indices,
                          const nd_array<T> &values) -> void;
template <typename T>
auto assign_masked_scalar(nd_array<T> &target,
                          const nd_array<std::int8_t> &mask, T scalar) -> void;
template <typename T>
auto assign_masked_array(nd_array<T> &target, const nd_array<std::int8_t> &mask,
                         const nd_array<T> &values) -> void;

#define TF_CPP_EXTERN_ASSIGN(T)                                                \
  extern template auto assign_scalar<T>(nd_array<T> &, T) -> void;             \
  extern template auto assign_array<T>(nd_array<T> &, const nd_array<T> &)     \
      -> void;                                                                 \
  extern template auto assign_indexed_scalar<T>(                               \
      nd_array<T> &, const nd_array<std::int32_t> &, T) -> void;               \
  extern template auto assign_indexed_array<T>(nd_array<T> &,                  \
                                               const nd_array<std::int32_t> &, \
                                               const nd_array<T> &) -> void;   \
  extern template auto assign_masked_scalar<T>(                                \
      nd_array<T> &, const nd_array<std::int8_t> &, T) -> void;                \
  extern template auto assign_masked_array<T>(nd_array<T> &,                   \
                                              const nd_array<std::int8_t> &,   \
                                              const nd_array<T> &) -> void

TF_CPP_EXTERN_ASSIGN(std::int8_t);
TF_CPP_EXTERN_ASSIGN(std::int32_t);
TF_CPP_EXTERN_ASSIGN(float);
TF_CPP_EXTERN_ASSIGN(double);

#undef TF_CPP_EXTERN_ASSIGN

} // namespace tf::cpp
