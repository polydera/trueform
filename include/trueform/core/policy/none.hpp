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
#include "../none.hpp"

namespace tf {

/// @ingroup core_policies
/// @brief Pipe operator for none tag (no-op).
///
/// Returns the input unchanged.
template <typename U> auto operator|(U &&u, none_t) -> U && {
  return static_cast<U &&>(u);
}

/// @ingroup core_policies
/// @brief Create none tag operator.
///
/// Allows uniform tagging syntax when no tag is needed.
///
/// @return The none tag.
inline auto tag(none_t) -> none_t { return none; }

/// @ingroup core_policies
/// @brief Tag with none (returns input unchanged).
/// @overload
template <typename U> auto tag(none_t, U &&u) -> U && {
  return static_cast<U &&>(u);
}

} // namespace tf
