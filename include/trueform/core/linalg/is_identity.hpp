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
#include "./identity.hpp"
#include "./identity_frame.hpp"

namespace tf::linalg {
namespace implementation {
template <typename T, std::size_t Dims>
auto is_identity_impl(const tf::linalg::identity<T, Dims> *) -> std::true_type;
auto is_identity_impl(const void *) -> std::false_type;

template <typename T, std::size_t Dims>
auto is_identity_impl(const tf::linalg::identity_frame<T, Dims> *)
    -> std::true_type;
} // namespace implementation

template <typename T>
inline constexpr bool is_identity = decltype(implementation::is_identity_impl(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;
} // namespace tf::linalg
