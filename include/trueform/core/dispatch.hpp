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

#include <type_traits>

namespace tf::core {

/// Tag type for dispatching based on element type without runtime access
template <typename T>
struct dispatch_t {};

/// Create a dispatch tag from a range's element type (compile-time only)
template <typename Range>
constexpr auto dispatch_element(const Range &)
    -> dispatch_t<std::decay_t<decltype(std::declval<Range>()[0])>> {
  return {};
}

} // namespace tf::core
