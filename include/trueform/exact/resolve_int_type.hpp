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

#include "../core/none.hpp"
#include "./int32.hpp"
#include "./int64.hpp"
#include <type_traits>

namespace tf::exact {

namespace detail {

template <typename CoordType> struct default_int_for {
  using type = int32;
};
template <> struct default_int_for<float> {
  using type = int32;
};
template <> struct default_int_for<double> {
  using type = int64;
};

} // namespace detail

/// Resolves a user-supplied `Int` against an input coordinate type.
///
/// `tf::none_t` → int32 for float input, int64 for double input,
/// int32 fallback for any other coord type. Any concrete int type is
/// returned as-is.
template <typename Int, typename CoordType>
using resolve_int_type = std::conditional_t<
    std::is_same_v<Int, tf::none_t>,
    typename detail::default_int_for<std::decay_t<CoordType>>::type, Int>;

} // namespace tf::exact
