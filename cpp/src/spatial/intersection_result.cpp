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
#include "trueform/cpp/spatial/intersection_result.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>

namespace tf::cpp {

intersection_result::intersection_result(bool value) : _value(value) {}

intersection_result::intersection_result(nd_array<std::int8_t> value)
    : _value(std::move(value)) {
  const auto &batch = std::get<nd_array<std::int8_t>>(_value);
  if (!batch.is_valid() || batch.ndim() != 1)
    throw std::invalid_argument(
        "intersection_result: batch must be a valid one-dimensional array");
}

auto intersection_result::is_scalar() const -> bool {
  return std::holds_alternative<bool>(_value);
}

auto intersection_result::is_batch() const -> bool {
  return std::holds_alternative<nd_array<std::int8_t>>(_value);
}

auto intersection_result::scalar() const -> bool {
  if (!is_scalar())
    throw std::logic_error("intersection_result: result is not scalar");
  return std::get<bool>(_value);
}

auto intersection_result::batch() const -> nd_array<std::int8_t> {
  if (!is_batch())
    throw std::logic_error("intersection_result: result is not a batch");
  return std::get<nd_array<std::int8_t>>(_value).shallow_copy();
}

} // namespace tf::cpp
