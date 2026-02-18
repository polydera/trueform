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

#include <limits>
#include <type_traits>

namespace tf {

/// @ingroup core
/// @brief Type-safe handle wrapping an integral ID.
///
/// Provides a strongly-typed wrapper around integral identifiers to prevent
/// accidental mixing of different ID domains. The invalid/sentinel value is
/// `std::numeric_limits<T>::max()` for unsigned types and `-1` for signed types.
///
/// @tparam T The underlying integral type.
template <typename T> class id_handle {
  static_assert(std::is_integral_v<T>,
                "Underlying id type must be an integral.");

protected:
  static constexpr T _no_id =
      std::is_unsigned_v<T> ? std::numeric_limits<T>::max() : T(-1);

public:
  id_handle() = default;
  explicit id_handle(T id) : _id{id} {}

  static auto invalid() -> id_handle { return id_handle{_no_id}; }

  auto id() const -> T { return _id; }
  auto is_valid() const -> bool { return _id != _no_id; }
  auto reset(T id = _no_id) -> void { _id = id; }

  template <typename U>
  auto operator==(const id_handle<U> &rhs) const -> bool {
    return _id == rhs.id();
  }
  template <typename U>
  auto operator!=(const id_handle<U> &rhs) const -> bool {
    return _id != rhs.id();
  }
  template <typename U>
  auto operator<(const id_handle<U> &rhs) const -> bool {
    return _id < rhs.id();
  }
  template <typename U>
  auto operator>(const id_handle<U> &rhs) const -> bool {
    return _id > rhs.id();
  }
  template <typename U>
  auto operator<=(const id_handle<U> &rhs) const -> bool {
    return _id <= rhs.id();
  }
  template <typename U>
  auto operator>=(const id_handle<U> &rhs) const -> bool {
    return _id >= rhs.id();
  }

private:
  T _id;
};

} // namespace tf
