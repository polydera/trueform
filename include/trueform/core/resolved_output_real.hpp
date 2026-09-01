/*
 * Copyright (c) 2026 XLAB
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

#include "./none.hpp"

#include <type_traits>

namespace tf {

/// @ingroup core
/// @brief The output coordinate type an entry point emits in.
///
/// `Request` is what the caller asked for and may be @ref tf::none_t, which
/// asks for the input's own type. The resolved type states the law once: an
/// output coordinate is floating-point or integral, and an integral input
/// carries no converter back to the reals, so it cannot answer in them.
template <typename Request, typename InputReal> struct resolved_output_real {
  using type =
      std::conditional_t<std::is_same_v<Request, tf::none_t>, InputReal,
                         Request>;
  static_assert(std::is_floating_point_v<type> || std::is_integral_v<type>,
                "Output coordinate type must be floating-point or integral");
  static_assert(!std::is_integral_v<InputReal> ||
                    !std::is_floating_point_v<type>,
                "Integer input cannot produce floating-point output");
};

/// @ingroup core
/// @brief The resolved output coordinate type; see @ref
///        tf::resolved_output_real.
template <typename Request, typename InputReal>
using resolved_output_real_t =
    typename resolved_output_real<Request, InputReal>::type;

} // namespace tf
