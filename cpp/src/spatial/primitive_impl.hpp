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

#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace tf::cpp {
namespace {

auto require_shape(const tf::small_vector<int, 3> &shape,
                   std::initializer_list<int> expected) -> bool {
  if (shape.size() != expected.size())
    return false;
  auto value = shape.begin();
  auto required = expected.begin();
  for (; required != expected.end(); ++required, ++value)
    if (*required >= 0 && *value != *required)
      return false;
  return true;
}

struct primitive_metadata {
  primitive_cardinality cardinality;
  int count;
  int polygon_vertex_count;
  std::size_t element_stride;
};

template <std::size_t Dims>
auto validate(primitive_kind kind, const tf::small_vector<int, 3> &shape)
    -> primitive_metadata {
  const auto dims = static_cast<int>(Dims);
  const auto single_or_batch_vector = [&]() -> primitive_metadata {
    if (require_shape(shape, {dims}))
      return {primitive_cardinality::single, 1, 0, Dims};
    if (require_shape(shape, {-1, dims}))
      return {primitive_cardinality::batch, shape[0], 0, Dims};
    if constexpr (Dims == 3)
      throw std::invalid_argument(
          "primitive: point/vector shape must be [3] or [N, 3]");
    else
      throw std::invalid_argument(
          "primitive: point/vector shape must be [2] or [N, 2]");
  };

  const auto single_or_batch_pair = [&]() -> primitive_metadata {
    if (require_shape(shape, {2, dims}))
      return {primitive_cardinality::single, 1, 0, 2 * Dims};
    if (require_shape(shape, {-1, 2, dims}))
      return {primitive_cardinality::batch, shape[0], 0, 2 * Dims};
    if constexpr (Dims == 3)
      throw std::invalid_argument(
          "primitive: pair shape must be [2, 3] or [N, 2, 3]");
    else
      throw std::invalid_argument(
          "primitive: pair shape must be [2, 2] or [N, 2, 2]");
  };

  switch (kind) {
  case primitive_kind::point:
  case primitive_kind::vector:
    return single_or_batch_vector();
  case primitive_kind::segment:
  case primitive_kind::ray:
  case primitive_kind::line:
  case primitive_kind::aabb:
    return single_or_batch_pair();
  case primitive_kind::triangle:
    if (require_shape(shape, {3, dims}))
      return {primitive_cardinality::single, 1, 0, 3 * Dims};
    if (require_shape(shape, {-1, 3, dims}))
      return {primitive_cardinality::batch, shape[0], 0, 3 * Dims};
    if constexpr (Dims == 3)
      throw std::invalid_argument(
          "primitive: triangle shape must be [3, 3] or [N, 3, 3]");
    else
      throw std::invalid_argument(
          "primitive: triangle shape must be [3, 2] or [N, 3, 2]");
  case primitive_kind::plane:
    if constexpr (Dims == 3) {
      if (require_shape(shape, {4}))
        return {primitive_cardinality::single, 1, 0, 4};
      if (require_shape(shape, {-1, 4}))
        return {primitive_cardinality::batch, shape[0], 0, 4};
      throw std::invalid_argument(
          "primitive: plane shape must be [4] or [N, 4]");
    } else {
      throw std::invalid_argument("primitive: plane requires Dims=3");
    }
  case primitive_kind::polygon:
    if (require_shape(shape, {-1, dims}) && shape[0] >= 3)
      return {primitive_cardinality::single, 1, shape[0],
              static_cast<std::size_t>(shape[0]) * Dims};
    if (require_shape(shape, {-1, -1, dims}) && shape[1] >= 3)
      return {primitive_cardinality::batch, shape[0], shape[1],
              static_cast<std::size_t>(shape[1]) * Dims};
    if constexpr (Dims == 3)
      throw std::invalid_argument(
          "primitive: polygon shape must be [V, 3] or [N, V, 3] with V >= 3");
    else
      throw std::invalid_argument(
          "primitive: polygon shape must be [V, 2] or [N, V, 2] with V >= 3");
  }
  throw std::invalid_argument("primitive: unknown primitive kind");
}

} // namespace

template <typename Real, std::size_t Dims>
primitive<Real, Dims>::primitive(primitive_kind kind, nd_array<Real> data)
    : _kind(kind), _cardinality(primitive_cardinality::single), _count(0),
      _polygon_vertex_count(0), _element_stride(0), _data(std::move(data)) {
  if (!_data.is_valid())
    throw std::invalid_argument("primitive: data is not valid");
  const auto metadata = validate<Dims>(kind, _data.raw_shape());
  _cardinality = metadata.cardinality;
  _count = metadata.count;
  _polygon_vertex_count = metadata.polygon_vertex_count;
  _element_stride = metadata.element_stride;
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::kind() const -> primitive_kind {
  return _kind;
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::cardinality() const -> primitive_cardinality {
  return _cardinality;
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::is_batch() const -> bool {
  return _cardinality == primitive_cardinality::batch;
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::count() const -> int {
  return _count;
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::polygon_vertex_count() const -> int {
  return _polygon_vertex_count;
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::element_stride() const -> std::size_t {
  return _element_stride;
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::data() const -> nd_array<Real> {
  return _data.shallow_copy();
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::at(int index) const -> primitive {
  if (is_batch())
    return primitive(_kind, _data.row(index));
  if (index != 0)
    throw std::out_of_range("primitive: index out of range");
  return shallow_copy();
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::slice(int start, int end) const -> primitive {
  if (!is_batch())
    throw std::logic_error("primitive: slice requires a batch");
  return primitive(_kind, _data.slice(start, end));
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::shallow_copy() const -> primitive {
  return primitive(_kind, _data.shallow_copy());
}

template <typename Real, std::size_t Dims>
auto primitive<Real, Dims>::deep_copy() const -> primitive {
  return primitive(_kind, _data.deep_copy());
}

} // namespace tf::cpp
