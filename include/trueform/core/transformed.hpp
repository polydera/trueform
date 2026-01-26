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
#include "./aabb_like.hpp"
#include "./base/aabb.hpp"
#include "./base/line.hpp"
#include "./base/obb_impl.hpp"
#include "./base/obbrss_impl.hpp"
#include "./base/plane.hpp"
#include "./base/poly.hpp"
#include "./base/ray.hpp"
#include "./base/rss_impl.hpp"
#include "./base/seg.hpp"
#include "./frame_like.hpp"
#include "./is_transformable.hpp"
#include "./linalg/is_identity.hpp"
#include "./linalg/transpose.hpp"
#include "./line_like.hpp"
#include "./obb_like.hpp"
#include "./obbrss_like.hpp"
#include "./point_like.hpp"
#include "./policy/buffer.hpp"
#include "./policy/id.hpp"
#include "./policy/ids.hpp"
#include "./policy/indices.hpp"
#include "./policy/normal.hpp"
#include "./policy/normals.hpp"
#include "./policy/plane.hpp"
#include "./policy/state.hpp"
#include "./policy/states.hpp"
#include "./polygon.hpp"
#include "./ray_like.hpp"
#include "./rss_like.hpp"
#include "./segment.hpp"
#include "./transformation.hpp"
#include "./tuple.hpp"
#include "./unit_vector_like.hpp"
#include "./vector_like.hpp"
#include "./views/indirect_range.hpp"
#include "./zip_range.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Apply a transformation to a geometric primitive.
///
/// Returns a new primitive with the transformation applied. Supports points,
/// vectors, unit vectors, lines, rays, planes, segments, polygons, AABBs,
/// OBBs, RSS, transformations, and frames. Identity transformations are
/// optimized to return the input unchanged.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The primitive's storage policy.
/// @tparam U The transformation's storage policy.
/// @param _this The primitive to transform.
/// @param transform The transformation to apply.
/// @return The transformed primitive.
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const point_like<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @brief Apply a frame transformation to a geometric primitive.
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const point_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const vector_like<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const vector_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const plane_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const unit_vector_like<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>) {
    auto out = wrap_like(_this, transformed(unwrap(_this), transform));
    tf::core::normalize(out);
    return out;
  } else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const unit_vector_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>) {
    auto out = wrap_like(_this, transformed(unwrap(_this), transform));
    tf::core::normalize(out);
    return out;
  } else
    return _this;
}

/// @ingroup core_primitives
/// @brief Transform a normal vector correctly using the inverse transpose.
///
/// Normal vectors require special handling under non-uniform scaling.
/// This function uses the inverse-transpose of the transformation matrix
/// to correctly transform normals.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The normal's storage policy.
/// @tparam U The frame's storage policy.
/// @param _this The unit normal to transform.
/// @param frame The frame transformation to apply.
/// @return The correctly transformed unit normal.
template <std::size_t Dims, typename Policy, typename U>
auto transformed_normal(const unit_vector_like<Dims, Policy> &_this,
                        const frame_like<Dims, U> &frame) {
  if constexpr (!linalg::is_identity<U>) {
    auto inv_frame = tf::make_frame_like(
        tf::linalg::make_transpose_view(frame.inverse_transformation()),
        tf::linalg::make_transpose_view(frame.transformation()));
    auto out = transformed(_this, inv_frame);
    tf::core::normalize(out);
    return out;
  } else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const ray_like<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const ray_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const line_like<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const line_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const aabb_like<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const aabb_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const segment<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const segment<Dims, Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const polygon<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const polygon<Dims, Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const transformation_like<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>) {
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
  } else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const frame_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &frame) {
  if constexpr (!linalg::is_identity<U>) {
    return tf::make_frame_like(
        transformed(_this.transformation(), frame.transformation()),
        transformed(frame.inverse_transformation(),
                    _this.inverse_transformation()));
  } else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <typename... Ts, std::size_t Dims, typename U>
auto transformed(const tuple<Ts...> &_this,
                 const transformation_like<Dims, U> &transform) {
  return std::apply(
      [&](const auto &...ts) {
        auto make_transformed = [&](const auto &x) -> decltype(auto) {
          if constexpr (core::is_transformable<decltype(x),
                                               frame_like<Dims, U>>)
            return transformed(x, transform);
          else
            return x;
        };
        return tf::forward_lref_as_tuple(make_transformed(ts)...);
      },
      _this);
}

/// @ingroup core_primitives
/// @overload
template <typename... Ts, std::size_t Dims, typename U>
auto transformed(const tuple<Ts...> &_this,
                 const frame_like<Dims, U> &transform) {
  return std::apply(
      [&](const auto &...ts) {
        auto make_transformed = [&](const auto &x) -> decltype(auto) {
          if constexpr (core::is_transformable<decltype(x),
                                               frame_like<Dims, U>>)
            return transformed(x, transform);
          else
            return x;
        };
        return tf::forward_lref_as_tuple(make_transformed(ts)...);
      },
      _this);
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const rss_like<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const rss_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &frame) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), frame));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const obb_like<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const obb_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &frame) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), frame));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const obbrss_like<Dims, Policy> &_this,
                 const frame_like<Dims, U> &frame) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), frame));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <std::size_t Dims, typename Policy, typename U>
auto transformed(const obbrss_like<Dims, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (!linalg::is_identity<U>)
    return wrap_like(_this, transformed(unwrap(_this), transform));
  else
    return _this;
}

/// @ingroup core_primitives
/// @overload
template <typename T, std::size_t Dims, typename U>
auto transformed(const std::array<std::array<T, Dims>, Dims> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (linalg::is_identity<U>)
    return _this;
  else {
    std::array<std::array<T, Dims>, Dims> out;
    for (std::size_t i = 0; i < Dims; ++i) {
      for (std::size_t j = 0; j < Dims; ++j) {
        out[i][j] = 0;
        for (std::size_t k = 0; k < Dims; ++k) {
          out[i][j] += transform(i, k) * _this(k, j);
        }
      }
    }
    return out;
  }
}

/// @ingroup core_primitives
/// @overload
template <typename T, std::size_t Dims, typename U>
auto transformed(const std::array<std::array<T, Dims>, Dims> &_this,
                 const frame_like<Dims, U> &transform) {
  return transformed(_this, transform.transformation());
}

} // namespace tf

namespace tf::core {

template <std::size_t N, typename Range, typename U>
auto transformed_range(Range &&r, const U &transform) {
  static_assert(N != tf::dynamic_size);
  if constexpr (tf::core::is_transformable<decltype(r[0]), U>) {
    std::array<decltype(transformed(r[0], transform)), N> out;
    for (std::size_t i = 0; i < N; ++i)
      out[i] = transformed(r[i], transform);
    return out;
  } else
    return r;
}

template <std::size_t N, typename... Ranges, typename U>
auto transformed_range(const zip_range<Ranges...> &rs, const U &transform) {
  static_assert(N != tf::dynamic_size);
  return std::apply(
      [&transform](auto &&...rs) {
        return tf::core::make_zip_range(transformed_range<N>(rs, transform)...);
      },
      rs.ranges());
}

template <typename T, std::size_t Size, typename U>
auto transformed(const pt<T, Size> &data,
                 const transformation_like<Size, U> &transform) {
  pt<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  transform.transform_point(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed(const pt<T, Size> &data, const frame_like<Size, U> &frame) {
  pt<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  frame.transformation().transform_point(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed(const pt_view<T, Size> &data,
                 const transformation_like<Size, U> &transform) {
  pt<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  transform.transform_point(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed(const pt_view<T, Size> &data,
                 const frame_like<Size, U> &frame) {
  pt<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  frame.transformation().transform_point(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed(const vec<T, Size> &data,
                 const transformation_like<Size, U> &transform) {
  vec<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  transform.transform_vector(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed(const vec<T, Size> &data, const frame_like<Size, U> &frame) {
  vec<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  frame.transformation().transform_vector(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed(const vec_view<T, Size> &data,
                 const transformation_like<Size, U> &transform) {
  vec<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  transform.transform_vector(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed(const vec_view<T, Size> &data,
                 const frame_like<Size, U> &frame) {
  vec<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  frame.transformation().transform_vector(data.data(), ptr);
  return out;
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const ray<Dims, Policy0, Policy1> &ray,
                 const transformation_like<Dims, U> &transform) {
  return core::make_ray(transformed(ray.origin, transform),
                        transformed(ray.direction, transform));
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const ray<Dims, Policy0, Policy1> &ray,
                 const frame_like<Dims, U> &transform) {
  return core::make_ray(transformed(ray.origin, transform),
                        transformed(ray.direction, transform));
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const line<Dims, Policy0, Policy1> &line,
                 const transformation_like<Dims, U> &transform) {
  return core::make_ray(transformed(line.origin, transform),
                        transformed(line.direction, transform));
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const line<Dims, Policy0, Policy1> &line,
                 const frame_like<Dims, U> &transform) {
  return core::make_ray(transformed(line.origin, transform),
                        transformed(line.direction, transform));
}

template <std::size_t Dims, typename Policy, typename U>
auto transformed(const plane<Dims, Policy> &plane,
                 const frame_like<Dims, U> &frame) {
  auto normal = transformed_normal(plane.normal, frame);
  tf::coordinate_type<U> d = plane.d;
  // we have a translation
  if constexpr (Dims + 1 ==
                std::decay_t<decltype(frame.transformation())>::n_columns) {
    tf::vector<tf::coordinate_type<U>, Dims> translation;
    for (std::size_t i = 0; i < Dims; ++i)
      translation[i] = frame.transformation()(i, Dims);
    d -= tf::dot(normal, translation);
  }
  return core::make_plane(normal, d);
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const aabb<Dims, Policy0, Policy1> &_this,
                 const transformation_like<Dims, U> &transform) {
  using real_t = decltype(_this.max[0] * transform(0, 0));

  aabb<Dims, pt<real_t, Dims>, pt<real_t, Dims>> out;
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

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const aabb<Dims, Policy0, Policy1> &_this,
                 const frame_like<Dims, U> &frame) {
  return transformed(_this, frame.transformation());
}

template <std::size_t Dims, typename Policy, typename U>
auto transformed(const seg<Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  using pt_t = decltype(transformed(_this[0], transform));
  return core::make_seg(std::array<pt_t, 2>{transformed(_this[0], transform),
                                            transformed(_this[1], transform)});
}

template <std::size_t Dims, typename Policy, typename U>
auto transformed(const seg<Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  using pt_t = decltype(transformed(_this[0], transform));
  return core::make_seg(std::array<pt_t, 2>{transformed(_this[0], transform),
                                            transformed(_this[1], transform)});
}

template <std::size_t V, std::size_t Dims, typename Policy, typename U>
auto transformed(const poly<V, Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  static_assert(V != tf::dynamic_size);
  using pt_t = decltype(transformed(_this[0], transform));
  std::array<pt_t, V> out;
  for (std::size_t i = 0; i < V; ++i)
    out[i] = transformed(_this[i], transform);
  return core::make_poly(out);
}

template <std::size_t V, std::size_t Dims, typename Policy, typename U>
auto transformed(const poly<V, Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  static_assert(V != tf::dynamic_size);
  using pt_t = decltype(transformed(_this[0], transform));
  std::array<pt_t, V> out;
  for (std::size_t i = 0; i < V; ++i)
    out[i] = transformed(_this[i], transform);
  return core::make_poly(out);
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const rss<Dims, Policy0, Policy1> &_this,
                 const transformation_like<Dims, U> &transform) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  auto origin_out = transformed(_this.origin, transform);

  // Transform first axis as vector to get uniform scale factor
  using vec_like_t = tf::vector_like<Dims, Policy1>;
  auto first_scaled =
      transformed(static_cast<const vec_like_t &>(_this.axes[0]), transform);
  auto scale = first_scaled.length();
  auto inv_scale = T(1) / (scale + (scale == T(0)));
  first_scaled *= inv_scale;

  std::array<decltype(tf::make_unit_vector(tf::unsafe, first_scaled)), Dims>
      axes_out;
  axes_out[0] = tf::make_unit_vector(tf::unsafe, first_scaled);

  // Transform remaining axes (uniform scaling, reuse inv_scale)
  for (std::size_t i = 1; i < Dims; ++i) {
    auto scaled_axis =
        transformed(static_cast<const vec_like_t &>(_this.axes[i]), transform);
    scaled_axis *= inv_scale;
    axes_out[i] = tf::make_unit_vector(tf::unsafe, scaled_axis);
  }

  // Scale lengths and radius by uniform scale factor
  std::array<T, Dims - 1> length_out;
  for (std::size_t i = 0; i < Dims - 1; ++i) {
    length_out[i] = _this.length[i] * scale;
  }

  return core::make_rss(origin_out, axes_out, length_out, _this.radius * scale);
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const rss<Dims, Policy0, Policy1> &_this,
                 const frame_like<Dims, U> &frame) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  auto origin_out = transformed(_this.origin, frame);

  // Transform first axis as vector to get uniform scale factor
  using vec_like_t = tf::vector_like<Dims, Policy1>;
  auto first_scaled =
      transformed(static_cast<const vec_like_t &>(_this.axes[0]), frame);
  auto scale = first_scaled.length();
  auto inv_scale = T(1) / (scale + (scale == T(0)));
  first_scaled *= inv_scale;

  std::array<decltype(tf::make_unit_vector(tf::unsafe, first_scaled)), Dims>
      axes_out;
  axes_out[0] = tf::make_unit_vector(tf::unsafe, first_scaled);

  // Transform remaining axes (uniform scaling, reuse inv_scale)
  for (std::size_t i = 1; i < Dims; ++i) {
    auto scaled_axis =
        transformed(static_cast<const vec_like_t &>(_this.axes[i]), frame);
    scaled_axis *= inv_scale;
    axes_out[i] = tf::make_unit_vector(tf::unsafe, scaled_axis);
  }

  // Scale lengths and radius by uniform scale factor
  std::array<T, Dims - 1> length_out;
  for (std::size_t i = 0; i < Dims - 1; ++i) {
    length_out[i] = _this.length[i] * scale;
  }

  return core::make_rss(origin_out, axes_out, length_out, _this.radius * scale);
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const obb<Dims, Policy0, Policy1> &_this,
                 const transformation_like<Dims, U> &transform) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  auto origin_out = transformed(_this.origin, transform);

  // Transform axes as vectors to capture scaling
  using vec_like_t = tf::vector_like<Dims, Policy1>;
  auto first_scaled =
      transformed(static_cast<const vec_like_t &>(_this.axes[0]), transform);
  std::array<decltype(tf::make_unit_vector(tf::unsafe, first_scaled)), Dims>
      axes_out;
  std::array<T, Dims> extent_out;

  for (std::size_t i = 0; i < Dims; ++i) {
    auto scaled_axis =
        transformed(static_cast<const vec_like_t &>(_this.axes[i]), transform);
    auto scale = scaled_axis.length();
    scaled_axis /= scale + (scale == T(0));
    axes_out[i] = tf::make_unit_vector(tf::unsafe, scaled_axis);
    extent_out[i] = _this.extent[i] * scale;
  }

  return core::make_obb(origin_out, axes_out, extent_out);
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const obb<Dims, Policy0, Policy1> &_this,
                 const frame_like<Dims, U> &frame) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  auto origin_out = transformed(_this.origin, frame);

  // Transform axes as vectors to capture scaling
  using vec_like_t = tf::vector_like<Dims, Policy1>;
  auto first_scaled =
      transformed(static_cast<const vec_like_t &>(_this.axes[0]), frame);
  std::array<decltype(tf::make_unit_vector(tf::unsafe, first_scaled)), Dims>
      axes_out;
  std::array<T, Dims> extent_out;

  for (std::size_t i = 0; i < Dims; ++i) {
    auto scaled_axis =
        transformed(static_cast<const vec_like_t &>(_this.axes[i]), frame);
    auto scale = scaled_axis.length();
    scaled_axis /= scale + (scale == T(0));
    axes_out[i] = tf::make_unit_vector(tf::unsafe, scaled_axis);
    extent_out[i] = _this.extent[i] * scale;
  }

  return core::make_obb(origin_out, axes_out, extent_out);
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const obbrss<Dims, Policy0, Policy1> &_this,
                 const transformation_like<Dims, U> &transform) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  auto obb_origin_out = transformed(_this.obb_origin, transform);
  auto rss_origin_out = transformed(_this.rss_origin, transform);

  // Transform first axis as vector to get uniform scale factor
  using vec_like_t = tf::vector_like<Dims, Policy1>;
  auto first_scaled =
      transformed(static_cast<const vec_like_t &>(_this.axes[0]), transform);
  auto scale = first_scaled.length();
  auto inv_scale = T(1) / (scale + (scale == T(0)));
  first_scaled *= inv_scale;

  std::array<decltype(tf::make_unit_vector(tf::unsafe, first_scaled)), Dims>
      axes_out;
  axes_out[0] = tf::make_unit_vector(tf::unsafe, first_scaled);

  // Transform remaining axes (uniform scaling, reuse inv_scale)
  for (std::size_t i = 1; i < Dims; ++i) {
    auto scaled_axis =
        transformed(static_cast<const vec_like_t &>(_this.axes[i]), transform);
    scaled_axis *= inv_scale;
    axes_out[i] = tf::make_unit_vector(tf::unsafe, scaled_axis);
  }

  // Scale extents by uniform scale factor
  std::array<T, Dims> extent_out;
  for (std::size_t i = 0; i < Dims; ++i) {
    extent_out[i] = _this.extent[i] * scale;
  }

  // Scale lengths and radius by uniform scale factor
  std::array<T, Dims - 1> length_out;
  for (std::size_t i = 0; i < Dims - 1; ++i) {
    length_out[i] = _this.length[i] * scale;
  }

  return core::make_obbrss(obb_origin_out, rss_origin_out, axes_out, extent_out,
                           length_out, _this.radius * scale);
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename U>
auto transformed(const obbrss<Dims, Policy0, Policy1> &_this,
                 const frame_like<Dims, U> &frame) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  auto obb_origin_out = transformed(_this.obb_origin, frame);
  auto rss_origin_out = transformed(_this.rss_origin, frame);

  // Transform first axis as vector to get uniform scale factor
  using vec_like_t = tf::vector_like<Dims, Policy1>;
  auto first_scaled =
      transformed(static_cast<const vec_like_t &>(_this.axes[0]), frame);
  auto scale = first_scaled.length();
  auto inv_scale = T(1) / (scale + (scale == T(0)));
  first_scaled *= inv_scale;

  std::array<decltype(tf::make_unit_vector(tf::unsafe, first_scaled)), Dims>
      axes_out;
  axes_out[0] = tf::make_unit_vector(tf::unsafe, first_scaled);

  // Transform remaining axes (uniform scaling, reuse inv_scale)
  for (std::size_t i = 1; i < Dims; ++i) {
    auto scaled_axis =
        transformed(static_cast<const vec_like_t &>(_this.axes[i]), frame);
    scaled_axis *= inv_scale;
    axes_out[i] = tf::make_unit_vector(tf::unsafe, scaled_axis);
  }

  // Scale extents by uniform scale factor
  std::array<T, Dims> extent_out;
  for (std::size_t i = 0; i < Dims; ++i) {
    extent_out[i] = _this.extent[i] * scale;
  }

  // Scale lengths and radius by uniform scale factor
  std::array<T, Dims - 1> length_out;
  for (std::size_t i = 0; i < Dims - 1; ++i) {
    length_out[i] = _this.length[i] * scale;
  }

  return core::make_obbrss(obb_origin_out, rss_origin_out, axes_out, extent_out,
                           length_out, _this.radius * scale);
}

} // namespace tf::core
namespace tf::views {
template <typename Policy, std::size_t Dims, typename U>
auto transformed(const indirect_range<Policy> &_this,
                 const transformation_like<Dims, U> &transform) {
  return tf::tag_indices(
      _this.indices(),
      transformed(static_cast<const Policy &>(_this), transform));
}
template <typename Policy, std::size_t Dims, typename U>
auto transformed(const indirect_range<Policy> &_this,
                 const frame_like<Dims, U> &transform) {
  return tf::tag_indices(
      _this.indices(),
      transformed(static_cast<const Policy &>(_this), transform));
}
} // namespace tf::views
namespace tf::policy {
template <typename Index, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_state<Index, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (tf::core::is_transformable<decltype(_this.state()),
                                           transformation_like<Dims, U>>) {
    return tf::policy::tag_state_impl(transformed(_this.state(), transform),
                                      transformed(unwrap(_this), transform));
  } else
    return wrap_like(_this, transformed(unwrap(_this), transform));
}

template <typename Index, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_state<Index, Base> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (tf::core::is_transformable<decltype(_this.state()),
                                           frame_like<Dims, U>>) {
    return tag_state_impl(transformed(_this.state(), transform),
                          transformed(unwrap(_this), transform));
  } else
    return wrap_like(_this, transformed(unwrap(_this), transform));
}

template <typename Index, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_state_ptr<Index, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (tf::core::is_transformable<decltype(_this.state()),
                                           transformation_like<Dims, U>>) {
    return tag_state_impl(transformed(_this.state(), transform),
                          transformed(unwrap(_this), transform));
  } else
    return wrap_like(_this, transformed(unwrap(_this), transform));
}

template <typename Index, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_state_ptr<Index, Base> &_this,
                 const frame_like<Dims, U> &transform) {
  if constexpr (tf::core::is_transformable<decltype(_this.state()),
                                           frame_like<Dims, U>>) {
    return tag_state_impl(transformed(_this.state(), transform),
                          transformed(unwrap(_this), transform));
  } else
    return wrap_like(_this, transformed(unwrap(_this), transform));
}

template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_states<Range, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  auto states = core::transformed_range<tf::static_size_v<Base>>(_this.states(),
                                                                 transform);
  auto base = transformed(unwrap(_this), transform);
  return tag_states<decltype(states), decltype(base)>{states, base};
}
template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_states<Range, Base> &_this,
                 const frame_like<Dims, U> &transform) {

  auto states = core::transformed_range<tf::static_size_v<Base>>(_this.states(),
                                                                 transform);
  auto base = transformed(unwrap(_this), transform);
  return tag_states<decltype(states), decltype(base)>{states, base};
}

template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const zip_states<Range, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  auto states = core::transformed_range<tf::static_size_v<Base>>(_this.states(),
                                                                 transform);
  auto base = transformed(unwrap(_this), transform);
  return zip_states<decltype(states), decltype(base)>{states, base};
}

template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const zip_states<Range, Base> &_this,
                 const frame_like<Dims, U> &transform) {

  auto states = core::transformed_range<tf::static_size_v<Base>>(_this.states(),
                                                                 transform);
  auto base = transformed(unwrap(_this), transform);
  return zip_states<decltype(states), decltype(base)>{states, base};
}

template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_normals<Range, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  return transformed(unwrap(_this), transform);
}
template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_normals<Range, Base> &_this,
                 const frame_like<Dims, U> &frame) {
  auto inv_frame = tf::make_frame_like(
      tf::linalg::make_transpose_view(frame.inverse_transformation()),
      tf::linalg::make_transpose_view(frame.transformation()));
  auto normals = core::transformed_range<tf::static_size_v<Base>>(
      _this.normals(), inv_frame);
  auto base = transformed(unwrap(_this), frame);
  return tag_normals<decltype(normals), decltype(base)>{normals, base};
}

template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const zip_normals<Range, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  return transformed(unwrap(_this), transform);
}
template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const zip_normals<Range, Base> &_this,
                 const frame_like<Dims, U> &frame) {
  auto inv_frame = tf::make_frame_like(
      tf::linalg::make_transpose_view(frame.inverse_transformation()),
      tf::linalg::make_transpose_view(frame.transformation()));
  auto normals = core::transformed_range<tf::static_size_v<Base>>(
      _this.normals(), inv_frame);
  auto base = transformed(unwrap(_this), frame);
  return zip_normals<decltype(normals), decltype(base)>{normals, base};
}

template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_indices<Range, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  return wrap_like(_this, transformed(unwrap(_this), transform));
}

template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_indices<Range, Base> &_this,
                 const frame_like<Dims, U> &transform) {
  return wrap_like(_this, transformed(unwrap(_this), transform));
}

template <typename Index, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_id<Index, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  return wrap_like(_this, transformed(unwrap(_this), transform));
}
template <typename Index, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_id<Index, Base> &_this,
                 const frame_like<Dims, U> &transform) {
  return wrap_like(_this, transformed(unwrap(_this), transform));
}
template <typename Index, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_id_iter<Index, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  return wrap_like(_this, transformed(unwrap(_this), transform));
}
template <typename Index, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_id_iter<Index, Base> &_this,
                 const frame_like<Dims, U> &transform) {
  return wrap_like(_this, transformed(unwrap(_this), transform));
}
template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_ids<Range, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  return wrap_like(_this, transformed(unwrap(_this), transform));
}
template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const tag_ids<Range, Base> &_this,
                 const frame_like<Dims, U> &transform) {
  return wrap_like(_this, transformed(unwrap(_this), transform));
}
template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const zip_ids<Range, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  return wrap_like(_this, transformed(unwrap(_this), transform));
}
template <typename Range, typename Base, std::size_t Dims, typename U>
auto transformed(const zip_ids<Range, Base> &_this,
                 const frame_like<Dims, U> &transform) {
  return wrap_like(_this, transformed(unwrap(_this), transform));
}

template <std::size_t Dims, typename Policy, typename Base, typename U>
auto transformed(const tag_normal<Dims, Policy, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  auto base = transformed(unwrap(_this), transform);
  if constexpr (tf::core::is_poly<Base>) {
    return tf::tag_normal(tf::make_normal(base[0], base[1], base[2]),
                          std::move(base));
  } else
    return base;
}

template <std::size_t Dims, typename Policy, typename Base, typename U>
auto transformed(const tag_normal<Dims, Policy, Base> &_this,
                 const frame_like<Dims, U> &frame) {
  auto base = transformed(unwrap(_this), frame);
  return tf::tag_normal(tf::transformed_normal(_this.normal(), frame),
                        std::move(base));
}

template <std::size_t Dims, typename Policy, typename Base, typename U>
auto transformed(const tag_plane<Dims, Policy, Base> &_this,
                 const transformation_like<Dims, U> &transform) {
  auto base = transformed(unwrap(_this), transform);
  if constexpr (tf::core::is_poly<Base>) {
    return tf::tag_plane(
        tf::make_plane(tf::make_normal(base[0], base[1], base[2]), base[0]),
        std::move(base));
  } else
    return base;
}

template <std::size_t Dims, typename Policy, typename Base, typename U>
auto transformed(const tag_plane<Dims, Policy, Base> &_this,
                 const frame_like<Dims, U> &frame) {
  return tf::tag_plane(transformed(_this.plane(), frame),
                       transformed(unwrap(_this), frame));
}

template <typename T, std::size_t V, typename Policy, std::size_t Dims,
          typename U>
auto transformed(const tag_buffer<T, core::poly<V, Policy>> &_this,
                 const transformation_like<Dims, U> &transform) {
  if constexpr (V != tf::dynamic_size) {
    return wrap_like(_this, transformed(unwrap(_this), transform));
  } else {
    using transformed_t = decltype(transformed(_this[0], transform));
    static_assert(std::is_assignable_v<T &, transformed_t>,
                  "Buffer element type T must be assignable from the "
                  "transformed point type");
    auto &buf = _this.buffer();
    buf.allocate(_this.size());
    for (std::size_t i = 0; i < _this.size(); ++i)
      buf[i] = transformed(_this[i], transform);
    return core::make_poly<tf::dynamic_size>(tf::make_range(buf));
  }
}

template <typename T, std::size_t V, typename Policy, std::size_t Dims,
          typename U>
auto transformed(const tag_buffer<T, core::poly<V, Policy>> &_this,
                 const frame_like<Dims, U> &frame) {
  if constexpr (V != tf::dynamic_size) {
    return wrap_like(_this, transformed(unwrap(_this), frame));
  } else {
    using transformed_t = decltype(transformed(_this[0], frame));
    static_assert(std::is_assignable_v<T &, transformed_t>,
                  "Buffer element type T must be assignable from the "
                  "transformed point type");
    auto &buf = _this.buffer();
    buf.allocate(_this.size());
    for (std::size_t i = 0; i < _this.size(); ++i)
      buf[i] = transformed(_this[i], frame);
    return core::make_poly<tf::dynamic_size>(tf::make_range(buf));
  }
}
} // namespace tf::policy
