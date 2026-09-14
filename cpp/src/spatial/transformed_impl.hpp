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

#include "trueform/cpp/spatial/transformed.hpp"

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/views/sequence_range.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace tf::cpp {
namespace detail {

/// The batch cutoff a transform states for itself: one element is a matrix
/// applied to a point, which is the cheapest kernel this module has.
constexpr unsigned long transformed_parallel_threshold = 4096;

template <typename Real, std::size_t Dims>
auto transform_point(const Real *input, Real *output, const Real *matrix)
    -> void {
  constexpr auto matrix_stride = Dims + 1;
  for (std::size_t row = 0; row < Dims; ++row) {
    auto value = matrix[row * matrix_stride + Dims];
    for (std::size_t column = 0; column < Dims; ++column)
      value += matrix[row * matrix_stride + column] * input[column];
    output[row] = value;
  }
}

template <typename Real, std::size_t Dims>
auto transform_vector(const Real *input, Real *output, const Real *matrix)
    -> void {
  constexpr auto matrix_stride = Dims + 1;
  for (std::size_t row = 0; row < Dims; ++row) {
    Real value{0};
    for (std::size_t column = 0; column < Dims; ++column)
      value += matrix[row * matrix_stride + column] * input[column];
    output[row] = value;
  }
}

template <typename Real, std::size_t Dims>
auto transform_point_elements(int count, int vertex_count, const Real *input,
                              Real *output, const Real *matrix) -> void {
  const auto stride = static_cast<std::size_t>(vertex_count) * Dims;
  tf::parallel_for_each(tf::make_sequence_range(count), [=](int index) {
    const auto element_offset = static_cast<std::size_t>(index) * stride;
    for (int vertex = 0; vertex < vertex_count; ++vertex) {
      const auto vertex_offset = static_cast<std::size_t>(vertex) * Dims;
      transform_point<Real, Dims>(input + element_offset + vertex_offset,
                                  output + element_offset + vertex_offset,
                                  matrix);
    }
  }, tf::checked(transformed_parallel_threshold));
}

template <typename Real, std::size_t Dims>
auto transform_vector_elements(int count, const Real *input, Real *output,
                               const Real *matrix) -> void {
  tf::parallel_for_each(tf::make_sequence_range(count), [=](int index) {
    const auto offset = static_cast<std::size_t>(index) * Dims;
    transform_vector<Real, Dims>(input + offset, output + offset, matrix);
  }, tf::checked(transformed_parallel_threshold));
}

template <typename Real, std::size_t Dims>
auto transform_direction_elements(int count, const Real *input, Real *output,
                                  const Real *matrix) -> void {
  tf::parallel_for_each(tf::make_sequence_range(count), [=](int index) {
    constexpr auto stride = 2 * Dims;
    const auto offset = static_cast<std::size_t>(index) * stride;
    transform_point<Real, Dims>(input + offset, output + offset, matrix);
    transform_vector<Real, Dims>(input + offset + Dims, output + offset + Dims,
                                 matrix);

    Real length2{0};
    for (std::size_t axis = 0; axis < Dims; ++axis) {
      const auto coordinate = output[offset + Dims + axis];
      length2 += coordinate * coordinate;
    }
    if (length2 == Real{0})
      throw std::invalid_argument(
          "transformed: transformed direction has zero length");
    const auto inverse_length = Real{1} / std::sqrt(length2);
    for (std::size_t axis = 0; axis < Dims; ++axis)
      output[offset + Dims + axis] *= inverse_length;
  }, tf::checked(transformed_parallel_threshold));
}

template <typename Real, std::size_t Dims>
auto transform_aabb_elements(int count, const Real *input, Real *output,
                             const Real *matrix) -> void {
  tf::parallel_for_each(tf::make_sequence_range(count), [=](int index) {
    constexpr auto matrix_stride = Dims + 1;
    constexpr auto element_stride = 2 * Dims;
    const auto offset = static_cast<std::size_t>(index) * element_stride;
    for (std::size_t row = 0; row < Dims; ++row) {
      auto lower = matrix[row * matrix_stride + Dims];
      auto upper = lower;
      for (std::size_t column = 0; column < Dims; ++column) {
        const auto coefficient = matrix[row * matrix_stride + column];
        const auto transformed_min = coefficient * input[offset + column];
        const auto transformed_max =
            coefficient * input[offset + Dims + column];
        if (transformed_min < transformed_max) {
          lower += transformed_min;
          upper += transformed_max;
        } else {
          lower += transformed_max;
          upper += transformed_min;
        }
      }
      output[offset + row] = lower;
      output[offset + Dims + row] = upper;
    }
  }, tf::checked(transformed_parallel_threshold));
}

template <typename Real>
auto inverse_linear_3d(const Real *matrix) -> std::array<Real, 9> {
  const auto a00 = matrix[0];
  const auto a01 = matrix[1];
  const auto a02 = matrix[2];
  const auto a10 = matrix[4];
  const auto a11 = matrix[5];
  const auto a12 = matrix[6];
  const auto a20 = matrix[8];
  const auto a21 = matrix[9];
  const auto a22 = matrix[10];

  const auto determinant = a00 * (a11 * a22 - a12 * a21) -
                           a01 * (a10 * a22 - a12 * a20) +
                           a02 * (a10 * a21 - a11 * a20);
  if (determinant == Real{0})
    throw std::invalid_argument(
        "transformed: plane transformation matrix is singular");
  const auto inverse_determinant = Real{1} / determinant;
  return {(a11 * a22 - a12 * a21) * inverse_determinant,
          (a02 * a21 - a01 * a22) * inverse_determinant,
          (a01 * a12 - a02 * a11) * inverse_determinant,
          (a12 * a20 - a10 * a22) * inverse_determinant,
          (a00 * a22 - a02 * a20) * inverse_determinant,
          (a02 * a10 - a00 * a12) * inverse_determinant,
          (a10 * a21 - a11 * a20) * inverse_determinant,
          (a01 * a20 - a00 * a21) * inverse_determinant,
          (a00 * a11 - a01 * a10) * inverse_determinant};
}

template <typename Real>
auto transform_plane_elements(int count, const Real *input, Real *output,
                              const Real *matrix) -> void {
  const auto inverse = inverse_linear_3d(matrix);
  const std::array<Real, 3> translation{matrix[3], matrix[7], matrix[11]};
  tf::parallel_for_each(tf::make_sequence_range(count), [=](int index) {
    constexpr std::size_t stride = 4;
    const auto offset = static_cast<std::size_t>(index) * stride;
    std::array<Real, 3> normal{};
    for (std::size_t row = 0; row < 3; ++row)
      for (std::size_t column = 0; column < 3; ++column)
        normal[row] += inverse[column * 3 + row] * input[offset + column];

    Real length2{0};
    Real translated_d = input[offset + 3];
    for (std::size_t axis = 0; axis < 3; ++axis) {
      length2 += normal[axis] * normal[axis];
      translated_d -= normal[axis] * translation[axis];
    }
    if (length2 == Real{0})
      throw std::invalid_argument(
          "transformed: transformed plane normal has zero length");
    const auto inverse_length = Real{1} / std::sqrt(length2);
    for (std::size_t axis = 0; axis < 3; ++axis)
      output[offset + axis] = normal[axis] * inverse_length;
    output[offset + 3] = translated_d * inverse_length;
  }, tf::checked(transformed_parallel_threshold));
}

/// A placement is homogeneous, so its shape is `[Dims + 1, Dims + 1]` and the
/// message says so from the same number the check reads.
template <std::size_t Dims, typename Real>
auto require_transformation_matrix(const nd_array<Real> &matrix) -> void {
  const auto expected = static_cast<int>(Dims + 1);
  if (!matrix.is_valid())
    throw std::invalid_argument("transformed: matrix is not valid");
  if (matrix.ndim() != 2 || matrix.shape_at(0) != expected ||
      matrix.shape_at(1) != expected)
    throw std::invalid_argument("transformed: matrix must have shape [" +
                                std::to_string(expected) + ", " +
                                std::to_string(expected) + "]");
}

template <typename Real, std::size_t Dims>
auto require_transformable_storage(const primitive<Real, Dims> &value,
                                   const nd_array<Real> &data) -> void {
  if (!data.is_valid())
    throw std::invalid_argument("transformed: primitive data is not valid");
  const auto expected =
      static_cast<std::size_t>(value.is_batch() ? value.count() : 1) *
      value.element_stride();
  if (data.length() != expected)
    throw std::invalid_argument("transformed: malformed primitive storage");
}

} // namespace detail

template <typename Real, std::size_t Dims>
auto transformed(const primitive<Real, Dims> &value,
                 const nd_array<Real> &matrix) -> primitive<Real, Dims> {
  detail::require_transformation_matrix<Dims>(matrix);
  const auto input_array = value.data();
  detail::require_transformable_storage(value, input_array);

  tf::buffer<Real> output_buffer;
  output_buffer.allocate(input_array.length());
  const auto *input = input_array.raw_data();
  auto *output = output_buffer.data();
  const auto *transform = matrix.raw_data();
  const auto count = value.is_batch() ? value.count() : 1;

  switch (value.kind()) {
  case primitive_kind::point:
    detail::transform_point_elements<Real, Dims>(count, 1, input, output,
                                                 transform);
    break;
  case primitive_kind::vector:
    detail::transform_vector_elements<Real, Dims>(count, input, output,
                                                  transform);
    break;
  case primitive_kind::segment:
    detail::transform_point_elements<Real, Dims>(count, 2, input, output,
                                                 transform);
    break;
  case primitive_kind::triangle:
    detail::transform_point_elements<Real, Dims>(count, 3, input, output,
                                                 transform);
    break;
  case primitive_kind::ray:
  case primitive_kind::line:
    detail::transform_direction_elements<Real, Dims>(count, input, output,
                                                     transform);
    break;
  case primitive_kind::plane:
    if constexpr (Dims == 3) {
      detail::transform_plane_elements(count, input, output, transform);
    } else {
      throw std::invalid_argument("transformed: plane requires Dims=3");
    }
    break;
  case primitive_kind::aabb:
    detail::transform_aabb_elements<Real, Dims>(count, input, output,
                                                transform);
    break;
  case primitive_kind::polygon:
    detail::transform_point_elements<Real, Dims>(
        count, value.polygon_vertex_count(), input, output, transform);
    break;
  }

  return primitive<Real, Dims>(
      value.kind(), nd_array<Real>::from_buffer(std::move(output_buffer),
                                                input_array.raw_shape()));
}

} // namespace tf::cpp
