/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../frame_like.hpp"
#include "../indirect_range.hpp"
#include "../inject_id.hpp"
#include "../inject_ids.hpp"
#include "../inject_normal.hpp"
#include "../inject_plane.hpp"
#include "../point_like.hpp"
#include "../polygon.hpp"
#include "../transformation.hpp"
#include "../unit_vector_like.hpp"
#include "../vector_like.hpp"
#include "../vector_view.hpp"

namespace tf::implementation {
struct as_vector_t {};
struct as_point_t {};
template <typename T, std::size_t Size, typename U>
auto transformed_impl(as_point_t, const owned_data<T, Size> &data,
                      const transformation<U, Size> &transform) {
  owned_data<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  transform.transform_point(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed_impl(as_point_t, const borrowed_data<T, Size> &data,
                      const transformation<U, Size> &transform) {
  owned_data<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  transform.transform_point(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed_impl(as_point_t, const owned_data<T, Size> &data,
                      const frame_like<Size, U> &frame) {
  return transformed_impl(as_point_t{}, data, frame.transformation());
}

template <typename T, std::size_t Size, typename U>
auto transformed_impl(as_point_t, const borrowed_data<T, Size> &data,
                      const frame_like<Size, U> &frame) {
  return transformed_impl(as_point_t{}, data, frame.transformation());
}

template <typename T, std::size_t Size, typename U>
auto transformed_impl(as_vector_t, const owned_data<T, Size> &data,
                      const transformation<U, Size> &transform) {
  owned_data<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  transform.transform_vector(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed_impl(as_vector_t, const borrowed_data<T, Size> &data,
                      const transformation<U, Size> &transform) {
  owned_data<std::remove_const_t<T>, Size> out;
  auto ptr = out.data();
  transform.transform_vector(data.data(), ptr);
  return out;
}

template <typename T, std::size_t Size, typename U>
auto transformed_impl(as_vector_t, const owned_data<T, Size> &data,
                      const frame_like<Size, U> &frame) {
  return transformed_impl(as_vector_t{}, data, frame.transformation());
}

template <typename T, std::size_t Size, typename U>
auto transformed_impl(as_vector_t, const borrowed_data<T, Size> &data,
                      const frame_like<Size, U> &frame) {
  return transformed_impl(as_vector_t{}, data, frame.transformation());
}

template <std::size_t Dims, typename T, typename U>
auto transformed_impl(as_point_t, const point_like<Dims, T> &_this,
                      const transformation<U, Dims> &transform) {
  auto policy =
      transformed_impl(as_point_t{}, static_cast<const T &>(_this), transform);
  return point_like<Dims, decltype(policy)>{policy};
}

template <std::size_t Dims, typename T, typename U>
auto transformed_impl(as_point_t, const point_like<Dims, T> &_this,
                      const frame_like<Dims, U> &frame) {
  auto policy =
      transformed_impl(as_point_t{}, static_cast<const T &>(_this), frame);
  return point_like<Dims, decltype(policy)>{policy};
}

template <std::size_t Dims, typename T, typename U>
auto transformed_impl(as_vector_t as, const vector_like<Dims, T> &_this,
                      const transformation<U, Dims> &transform) {
  auto policy = transformed_impl(as, static_cast<const T &>(_this), transform);
  return vector_like<Dims, decltype(policy)>{policy};
}

template <std::size_t Dims, typename T, typename U>
auto transformed_impl(as_vector_t as, const vector_like<Dims, T> &_this,
                      const frame_like<Dims, U> &frame) {
  auto policy = transformed_impl(as, static_cast<const T &>(_this), frame);
  return vector_like<Dims, decltype(policy)>{policy};
}

template <std::size_t Dims, typename T, typename U>
auto transformed_impl(as_vector_t as, const unit_vector_like<Dims, T> &_this,
                      const transformation<U, Dims> &transform) {
  auto policy = transformed_impl(as, static_cast<const T &>(_this), transform);
  return unit_vector_like<Dims, decltype(policy)>{tf::unsafe, policy};
}

template <std::size_t Dims, typename T, typename U>
auto transformed_impl(as_vector_t as, const unit_vector_like<Dims, T> &_this,
                      const frame_like<Dims, U> &frame) {
  auto policy = transformed_impl(as, static_cast<const T &>(_this), frame);
  return unit_vector_like<Dims, decltype(policy)>{tf::unsafe, policy};
}

template <typename As, typename Id, typename Base, std::size_t Dims, typename U>
auto transformed_impl(As, const tf::inject_id_t<Id, Base> &_this,
                      const transformation<U, Dims> &transform) {
  return tf::inject_id(
      _this.id(),
      transformed_impl(As{}, static_cast<const Base &>(_this), transform));
}

template <typename As, typename Id, typename Base, std::size_t Dims, typename U>
auto transformed_impl(As, const tf::inject_id_t<Id, Base> &_this,
                      const frame_like<Dims, U> &frame) {
  return tf::inject_id(
      _this.id(),
      transformed_impl(As{}, static_cast<const Base &>(_this), frame));
}

template <typename As, typename Range, std::size_t Dims, typename U>
auto transformed_impl(As, const Range &_this,
                      const transformation<U, Dims> &transform) {
  constexpr std::size_t V = tf::static_size_v<Range>;
  static_assert(V != tf::dynamic_size);
  using el_t = decltype(transformed_impl(As{}, _this[0], transform));
  std::array<el_t, V> out;
  for (std::size_t i = 0; i < V; ++i)
    out[i] = transformed_impl(As{}, _this[i], transform);
  return out;
}

template <typename As, typename Range, std::size_t Dims, typename U>
auto transformed_impl(As, const Range &_this,
                      const frame_like<Dims, U> &frame) {
  constexpr std::size_t V = tf::static_size_v<Range>;
  static_assert(V != tf::dynamic_size);
  using el_t = decltype(transformed_impl(As{}, _this[0], frame));
  std::array<el_t, V> out;
  for (std::size_t i = 0; i < V; ++i)
    out[i] = transformed_impl(As{}, _this[i], frame);
  return out;
}

template <typename As, typename Range, typename Base, std::size_t Dims,
          typename U>
auto transformed_impl(As, const tf::inject_ids_t<Range, Base> &_this,
                      const transformation<U, Dims> &transform) {
  return tf::inject_ids(
      _this.ids(),
      transformed_impl(As{}, static_cast<const Base &>(_this), transform));
}

template <typename As, typename Range, typename Base, std::size_t Dims,
          typename U>
auto transformed_impl(As, const tf::inject_ids_t<Range, Base> &_this,
                      const frame_like<Dims, U> &frame) {
  return tf::inject_ids(
      _this.ids(),
      transformed_impl(As{}, static_cast<const Base &>(_this), frame));
}

template <typename As, typename Iterator, std::size_t N, std::size_t Dims,
          typename U>
auto transformed_impl(As, const tf::indirect_range<Iterator, N> &_this,
                      const transformation<U, Dims> &transform) {
  return tf::inject_ids(
      _this.ids(), transformed_impl(As{}, tf::make_range(_this), transform));
}

template <typename As, typename Iterator, std::size_t N, std::size_t Dims,
          typename U>
auto transformed_impl(As, const tf::indirect_range<Iterator, N> &_this,
                      const frame_like<Dims, U> &frame) {
  return tf::inject_ids(_this.ids(),
                        transformed_impl(As{}, tf::make_range(_this), frame));
}

template <typename As, typename T, std::size_t Dims, typename Base, typename U>
auto transformed_impl(As, const tf::inject_normal_t<T, Dims, Base> &_this,
                      const tf::transformation<U, Dims> &transform) {
  return transformed_impl(As{}, static_cast<const Base &>(_this), transform);
}

template <typename As, typename T, std::size_t Dims, typename Base, typename U>
auto transformed_impl(As, const tf::inject_normal_t<T, Dims, Base> &_this,
                      const frame_like<Dims, U> &frame) {
  return tf::inject_normal(
      transformed_impl(as_vector_t{}, _this.normal(),
                       frame.inverse_transformation()),
      transformed_impl(As{}, static_cast<const Base &>(_this), frame));
}

template <typename As, typename T, std::size_t Dims, typename Base, typename U>
auto transformed_impl(As, const tf::inject_plane_t<T, Dims, Base> &_this,
                      const tf::transformation<U, Dims> &transform) {
  return transformed_impl(As{}, static_cast<const Base &>(_this), transform);
}

template <typename As, typename T, std::size_t Dims, typename Base, typename U>
auto transformed_impl(As, const tf::inject_plane_t<T, Dims, Base> &_this,
                      const frame_like<Dims, U> &frame) {
  auto base = transformed_impl(As{}, static_cast<const Base &>(_this), frame);
  auto get_pt =
      [&]() -> decltype(auto) { // we are a point, not a polygon or segment
    if constexpr (std::is_fundamental_v<std::decay_t<decltype(base[0])>>)
      return tf::make_vector_view<Dims>(&base[0]);
    else
      return base[0];
  };
  if constexpr (tf::has_injected_normal<Base>) {
    return tf::inject_plane(tf::make_plane(base.normal(), get_pt()), base);
  } else {
    return tf::inject_plane(
        tf::make_plane(transformed_impl(as_vector_t{}, _this.normal(),
                                        frame.inverse_transformation()),
                       get_pt()),
        base);
  }
}

} // namespace tf::implementation
