/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./aabb.hpp"
#include "./identity_frame.hpp"
#include "./identity_transformation.hpp"
#include "./implementation/transformed.hpp"
#include "./inject_normal.hpp"
#include "./inject_plane.hpp"
#include "./line.hpp"
#include "./point_like.hpp"
#include "./polygon.hpp"
#include "./ray.hpp"
#include "./segment.hpp"
#include "./transformation.hpp"
#include "./unit_vector_like.hpp"
#include "./vector_like.hpp"

namespace tf {

template <typename T, typename U>
auto transformed(const T &_this, const identity_transformation_t<U> &)
    -> const T & {
  return _this;
}

template <typename T, typename U>
auto transformed(const T &_this, const identity_frame_t<U> &) -> const T & {
  return _this;
}

template <std::size_t Dims, typename T, typename U>
auto transformed(const point_like<Dims, T> &_this,
                 const transformation<U, Dims> &transform) {
  return transformed_impl(implementation::as_point_t{}, _this, transform);
}

template <std::size_t Dims, typename T, typename U>
auto transformed(const point_like<Dims, T> &_this,
                 const frame_like<Dims, U> &frame) {
  return transformed_impl(implementation::as_point_t{}, _this, frame);
}

template <std::size_t Dims, typename T, typename U>
auto transformed(const vector_like<Dims, T> &_this,
                 const transformation<U, Dims> &transform) {
  return transformed_impl(implementation::as_vector_t{}, _this, transform);
}

template <std::size_t Dims, typename T, typename U>
auto transformed(const vector_like<Dims, T> &_this,
                 const frame_like<Dims, U> &frame) {
  return transformed_impl(implementation::as_vector_t{}, _this, frame);
}

template <std::size_t Dims, typename T, typename U>
auto transformed(const unit_vector_like<Dims, T> &_this,
                 const transformation<U, Dims> &transform) {
  return transformed_impl(implementation::as_vector_t{}, _this, transform);
}

template <std::size_t Dims, typename T, typename U>
auto transformed(const unit_vector_like<Dims, T> &_this,
                 const frame_like<Dims, U> &frame) {
  return transformed_impl(implementation::as_vector_t{}, _this, frame);
}

/// @ingroup geometry
/// @brief Compose two affine transformations.
///
/// Given two affine transformations, this returns a new transformation
/// that is equivalent to applying `_this` first, then `transform`.
///
/// This performs standard matrix composition:
/// `result = transform * _this`
///
/// @tparam T Scalar type of the left-hand transformation.
/// @tparam Dims Dimensionality of the transformation.
/// @tparam U Scalar type of the right-hand transformation.
/// @param _this The transformation to apply first.
/// @param transform The transformation to apply second.
/// @return A new @ref tf::transformation representing the composed
/// transformation.
template <typename T, std::size_t Dims, typename U>
auto transformed(const transformation<T, Dims> &_this,
                 const transformation<U, Dims> &transform) {
  using real_t = decltype(_this(0, 0) * transform(0, 0));
  transformation<real_t, Dims> out_array;
  for (std::size_t i = 0; i < Dims; ++i) {
    for (std::size_t j = 0; j < Dims; ++j) {
      out_array(i, j) = 0;
      for (std::size_t k = 0; k < Dims; ++k) {
        out_array(i, j) += transform(i, k) * _this(k, j);
      }
    }
  }
  for (std::size_t i = 0; i < Dims; ++i) {
    out_array(i, Dims) = transform(i, Dims);
    for (std::size_t j = 0; j < Dims; ++j) {
      out_array(i, Dims) += transform(i, j) * _this(j, Dims);
    }
  }
  return out_array;
}

/// @ingroup geometry
/// @brief Apply a transformation to an axis-aligned bounding box (AABB).
///
/// Computes a conservative transformed bounding box by applying the given
/// affine transformation to the input AABB. The resulting box is guaranteed
/// to contain the transformed shape, even under rotation or shear.
///
///
/// @tparam T Scalar type of the input AABB.
/// @tparam Dims Dimensionality of the AABB and transformation.
/// @tparam U Scalar type of the transformation matrix.
/// @param _this The input AABB to be transformed.
/// @param transform The affine transformation to apply.
/// @return A new AABB that conservatively bounds the transformed input.
template <typename T, std::size_t Dims, typename U>
auto transformed(const aabb<T, Dims> &_this,
                 const transformation<U, Dims> &transform) {
  using real_t = decltype(_this.max[0] * transform(0, 0));
  aabb<real_t, Dims> out;
  auto size = Dims;
  for (decltype(size) i = 0; i < size; ++i) {
    out.max[i] = out.min[i] = transform(i, size);
    for (decltype(size) j = 0; j < size; ++j) {
      std::array<decltype(transform(i, j) * _this.min[j]), 2> vals{
          transform(i, j) * _this.min[j], transform(i, j) * _this.max[j]};
      auto mode = vals[0] > vals[1];
      out.min[i] += vals[mode];
      out.max[i] += vals[1 - mode];
    }
  }
  return out;
}

template <typename T, std::size_t Dims, typename U>
auto transformed(const aabb<T, Dims> &_this, const frame_like<Dims, U> &frame) {
  return transformed(_this, frame.transformation());
}

/// @ingroup geometry
/// @brief Apply a transformation to a ray
///
/// @tparam T Scalar type of the input AABB.
/// @tparam Dims Dimensionality of the AABB and transformation.
/// @tparam U Scalar type of the transformation matrix.
/// @param _this The input ray to be transformed.
/// @param transform The affine transformation to apply.
/// @return A new ray in the transformed space
template <typename T, std::size_t Dims, typename U>
auto transformed(const ray<T, Dims> &_this,
                 const transformation<U, Dims> &transform) {
  ray<T, Dims> out{transform(_this.origin), transform(_this.direction)};
  return out;
}

template <typename T, std::size_t Dims, typename U>
auto transformed(const ray<T, Dims> &_this, const frame_like<Dims, U> &frame) {
  return transformed(_this, frame.transformation());
}

/// @ingroup geometry
/// @brief Apply a transformation to a line
template <typename T, std::size_t Dims, typename U>
auto transformed(const line<T, Dims> &_this,
                 const transformation<U, Dims> &transform) {
  line<T, Dims> out{transform(_this.origin), transform(_this.direction)};
  return out;
}

template <typename T, std::size_t Dims, typename U>
auto transformed(const line<T, Dims> &_this, const frame_like<Dims, U> &frame) {
  return transformed(_this, frame.transformation());
}

/// @ingroup geometry
/// @brief Apply a transformation to a polygon
template <std::size_t V, typename T, std::size_t Dims, typename U>
auto transformed(const polygon<V, T> &_this,
                 const transformation<U, Dims> &transform) {
  auto out = tf::make_polygon<V>(transformed_impl(
      implementation::as_point_t{}, static_cast<const T &>(_this), transform));
  if constexpr (has_injected_plane<T>) {
    return tf::inject_plane(out);
  } else if constexpr (has_injected_normal<T>)
    return tf::inject_normal(out);
  else
    return out;
}

template <std::size_t V, typename T, std::size_t Dims, typename U>
auto transformed(const polygon<V, T> &_this, const frame_like<Dims, U> &frame) {
  return tf::make_polygon<V>(transformed_impl(
      implementation::as_point_t{}, static_cast<const T &>(_this), frame));
}

template <typename T, std::size_t Dims, typename U>
auto transformed(const polygon<tf::dynamic_size, T> &_this,
                 const transformation<U, Dims> &transform) = delete;

template <typename T, std::size_t Dims, typename U>
auto transformed(const polygon<tf::dynamic_size, T> &_this,
                 const frame_like<Dims, U> &frame) = delete;

/// @ingroup geometry
/// @brief Apply a transformation to a segment
template <typename Policy, std::size_t Dims, typename U>
auto transformed(const segment<Policy> &_this,
                 const transformation<U, Dims> &transform) {
  return tf::make_segment(transformed_impl(implementation::as_point_t{},
                                           static_cast<const Policy &>(_this),
                                           transform));
}
} // namespace tf
