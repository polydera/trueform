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

#include "./form.hpp"
#include <type_traits>

namespace tf {
namespace core::impl {

template <std::size_t Dims, typename Policy>
auto is_form_test(const form<Dims, Policy> *) -> std::true_type;
auto is_form_test(const void *) -> std::false_type;

} // namespace core::impl

/// @ingroup core_traits
/// @brief Check whether T is a form (polygons, points, segments, etc.).
template <typename T>
inline constexpr bool is_form = decltype(core::impl::is_form_test(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

} // namespace tf
