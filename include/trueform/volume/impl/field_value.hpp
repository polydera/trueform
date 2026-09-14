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
#include "../volume.hpp"
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace tf {
namespace volume_detail {

/// @brief The value a stored sample decides by: the one cast between the
/// field's storage type and the type a call decides and emits in.
///
/// A call has one such type, and the classifier and every crossing that
/// follows it read a sample through this, so no two of them can disagree about
/// which side of the isovalue a sample falls on. On an edge the classifier
/// called crossing, `s0 = v0 - iso` and `s1 = v1 - iso` therefore straddle
/// zero and `t = s0 / (s0 - s1)` lies in [0, 1] by construction.
///
/// Arithmetic downstream may be wider than @p Real: widening is exact and
/// preserves order, so it cannot reopen that disagreement.
template <typename Real, typename Scalar>
inline auto field_value(const Scalar &sample) -> Real {
  return static_cast<Real>(sample);
}

/// @brief The field's samples with the deciding type bound to them.
///
/// The form a pass holds instead of the storage pointer when the deciding type
/// has to travel with it through the passes below: every read is @ref
/// field_value, so no pass can reach the storage any other way.
template <typename Real, typename Scalar> class field_samples {
public:
  using value_type = Real;

  field_samples() = default;
  explicit field_samples(const Scalar *samples) : _samples(samples) {}

  auto operator[](std::ptrdiff_t i) const -> Real {
    return field_value<Real>(_samples[i]);
  }

  auto operator+(std::ptrdiff_t n) const -> field_samples {
    return field_samples(_samples + n);
  }

private:
  const Scalar *_samples = nullptr;
};

/// @brief The samples of @p vol, read in @p Real.
template <typename Real, typename Policy>
auto make_field_samples(const tf::volume<Policy> &vol) {
  using Scalar = std::remove_const_t<
      typename std::iterator_traits<decltype(vol.begin())>::value_type>;
  return field_samples<Real, Scalar>(vol.begin());
}

} // namespace volume_detail
} // namespace tf
