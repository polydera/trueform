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
#include <variant>

namespace tf::cpp {

/// @brief Scalar or batch result of a runtime intersection query.
class intersection_result {
  std::variant<bool, nd_array<std::int8_t>> _value;

public:
  explicit intersection_result(bool value);
  explicit intersection_result(nd_array<std::int8_t> value);

  auto is_scalar() const -> bool;
  auto is_batch() const -> bool;
  auto scalar() const -> bool;
  auto batch() const -> nd_array<std::int8_t>;
};

} // namespace tf::cpp
