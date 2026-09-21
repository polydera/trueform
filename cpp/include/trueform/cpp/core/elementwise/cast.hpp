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

template <typename From, typename To>
auto cast(const nd_array<From> &a) -> nd_array<To>;

#define TF_CPP_EXTERN_CAST(FROM, TO)                                           \
  extern template auto cast<FROM, TO>(const nd_array<FROM> &) -> nd_array<TO>

TF_CPP_EXTERN_CAST(std::int8_t, std::int32_t);
TF_CPP_EXTERN_CAST(std::int8_t, std::int64_t);
TF_CPP_EXTERN_CAST(std::int8_t, float);
TF_CPP_EXTERN_CAST(std::int8_t, double);
TF_CPP_EXTERN_CAST(std::int32_t, std::int8_t);
TF_CPP_EXTERN_CAST(std::int32_t, std::int64_t);
TF_CPP_EXTERN_CAST(std::int32_t, float);
TF_CPP_EXTERN_CAST(std::int32_t, double);
TF_CPP_EXTERN_CAST(std::int64_t, std::int8_t);
TF_CPP_EXTERN_CAST(std::int64_t, std::int32_t);
TF_CPP_EXTERN_CAST(std::int64_t, float);
TF_CPP_EXTERN_CAST(std::int64_t, double);
TF_CPP_EXTERN_CAST(float, std::int8_t);
TF_CPP_EXTERN_CAST(float, std::int32_t);
TF_CPP_EXTERN_CAST(float, std::int64_t);
TF_CPP_EXTERN_CAST(float, double);
TF_CPP_EXTERN_CAST(double, std::int8_t);
TF_CPP_EXTERN_CAST(double, std::int32_t);
TF_CPP_EXTERN_CAST(double, std::int64_t);
TF_CPP_EXTERN_CAST(double, float);

#undef TF_CPP_EXTERN_CAST

} // namespace tf::cpp
