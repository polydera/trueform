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

#include "./coordinate_dims.hpp"
#include <utility>

namespace tf {

/// @ingroup core_ranges
/// @brief Base class for ranges of primitives (polygons, points, segments).
///
/// Form provides the wrap/unwrap semantics that enable policy composition.
/// When policies are tagged onto a form, they are inserted into the policy
/// chain while preserving the outer wrapper type.
///
/// The ranges `polygons`, `points`, and `segments` inherit from form,
/// making them composable with spatial trees, frames, and other policies.
///
/// @tparam Dims The spatial dimensionality (typically 2 or 3).
/// @tparam Policy The underlying policy chain.
template <std::size_t Dims, typename Policy> struct form : public Policy {
  form(Policy &&policy) : Policy{std::move(policy)} {}
  form(const Policy &policy) : Policy{policy} {}

  friend auto unwrap(const form &f) -> const Policy & {
    return static_cast<const Policy &>(f);
  }

  friend auto unwrap(form &f) -> Policy & { return static_cast<Policy &>(f); }

  friend auto unwrap(form &&f) -> Policy && {
    return static_cast<Policy &&>(f);
  }
};

} // namespace tf
