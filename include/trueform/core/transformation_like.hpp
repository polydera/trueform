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
#include "./linalg/identity.hpp"
#include "./linalg/trans.hpp"
#include "./point.hpp"
#include "./vector.hpp"
#include <cstddef>
#include <utility>
namespace tf {

/// @ingroup core_primitives
/// @brief Base template for affine transformation matrix types.
///
/// Provides the common interface for transformations, including point and
/// vector transformation methods. The matrix is stored in row-major order
/// with dimensions `Dims x (Dims+1)` for affine transformations.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The storage policy.
template <std::size_t Dims, typename Policy>
struct transformation_like : Policy {
  transformation_like() = default;
  transformation_like(const Policy &policy) : Policy{policy} {}
  transformation_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::operator();
  using Policy::n_columns;
  using Policy::n_rows;

  template <typename T>
  operator tf::transformation_like<Dims, tf::linalg::trans<T, Dims>>() const {
    tf::transformation_like<Dims, tf::linalg::trans<T, Dims>> out;
    for (std::size_t i = 0; i < Dims; ++i) {
      for (std::size_t j = 0; j < Dims; ++j)
        out(i, j) = (*this)(i, j);
      if constexpr (Policy::n_columns == Dims + 1)
        out(i, Dims) = (*this)(i, Dims);
      else
        out(i, Dims) = 0;
    }
    return out;
  }

  template <typename T>
  auto operator=(const transformation_like<Dims, T> &other)
      -> std::enable_if_t<std::is_assignable_v<typename Policy::element_type &,
                                               typename T::element_type>,
                          transformation_like &> {
    static_assert(n_rows == T::n_rows && n_columns == T::n_columns);
    for (std::size_t i = 0; i < n_rows; ++i)
      for (std::size_t j = 0; j < n_columns; ++j)
        (*this)(i, j) = other(i, j);
    return *this;
  }

  constexpr auto rows() const -> std::size_t { return n_rows; }

  constexpr auto columns() const -> std::size_t { return n_columns; }

  template <typename Point0, typename Point1>
  auto transform_point(const Point0 &point, Point1 &out) const {
    for (std::size_t i = 0; i < Dims; ++i) {
      if constexpr (n_columns == Dims + 1)
        out[i] = (*this)(i, Dims);
      else
        out[i] = 0;
      for (std::size_t j = 0; j < Dims; ++j) {
        out[i] += point[j] * (*this)(i, j);
      }
    }
  }

  template <typename Point0> auto transform_point(const Point0 &point) const {
    tf::point<typename Policy::coordinate_type, Dims> out;
    transform_point(point, out);
    return out;
  }

  template <typename Point0, typename Point1>
  auto transform_vector(const Point0 &point, Point1 &out) const {
    for (std::size_t i = 0; i < Dims; ++i) {
      out[i] = 0;
      for (std::size_t j = 0; j < Dims; ++j) {
        out[i] += point[j] * (*this)(i, j);
      }
    }
  }

  template <typename Point0> auto transform_vector(const Point0 &point) const {
    tf::vector<typename Policy::coordinate_type, Dims> out;
    transform_vector(point, out);
    return out;
  }

  template <typename U>
  auto fill(const U *_ptr) -> std::enable_if_t<
      std::is_assignable_v<typename Policy::element_type &, U>> {
    for (std::size_t i = 0; i < rows(); ++i) {
      for (std::size_t j = 0; j < columns(); ++j)
        (*this)(i, j) = *_ptr++;
    }
  }

  template <typename U>
  auto fill(std::initializer_list<U> list) -> std::enable_if_t<
      std::is_assignable_v<typename Policy::element_type &, U>> {
    fill(list.begin());
  }
};

template <std::size_t Dims, typename T>
struct transformation_like<Dims, linalg::identity<T, Dims>>
    : linalg::identity<T, Dims> {
private:
  using Policy = linalg::identity<T, Dims>;

public:
  using Policy::Policy;
  using Policy::operator();
  using Policy::n_columns;
  using Policy::n_rows;

  constexpr auto rows() const -> std::size_t { return n_rows; }

  constexpr auto columns() const -> std::size_t { return n_columns; }

  template <typename Point0, typename Point1>
  auto transform_point(const Point0 &, Point1 &) const {}

  template <typename Point0, typename Point1>
  auto transform_vector(const Point0 &, Point1 &) const {}
};

/// @ingroup core_primitives
/// @brief An identity transformation that leaves points and vectors unchanged.
///
/// @tparam T The scalar type.
/// @tparam Dims The dimensionality.
template <typename T, std::size_t Dims>
using identity_transformation =
    transformation_like<Dims, linalg::identity<T, Dims>>;

} // namespace tf
