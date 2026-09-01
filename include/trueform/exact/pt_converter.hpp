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

#include "../core/aabb_from.hpp"
#include "../core/aabb_union.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/frame_of.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/polygons.hpp"
#include "../core/transformed.hpp"
#include "../spatial/aabb_from.hpp"
#include "../spatial/policy/tree.hpp"
#include "./pt_converter_identity.hpp"
#include "./resolve_int_type.hpp"
#include <cmath>
#include <limits>
#include <type_traits>

namespace tf::exact {

template <typename IntT, typename RealT, std::size_t Dims>
struct pt_converter_real {
  RealT scale;
  tf::point<RealT, Dims> offset;

  template <typename P>
  auto operator()(const P &p) const -> tf::point<IntT, Dims> {
    tf::point<IntT, Dims> out;
    for (std::size_t i = 0; i < Dims; ++i)
      out[i] = static_cast<IntT>(
          std::round((static_cast<RealT>(p[i]) - offset[i]) * scale));
    return out;
  }

  auto deconvert(const tf::point<IntT, Dims> &ip) const
      -> tf::point<RealT, Dims> {
    tf::point<RealT, Dims> out;
    for (std::size_t i = 0; i < Dims; ++i)
      out[i] = static_cast<RealT>(ip[i]) / scale + offset[i];
    return out;
  }

  auto convert_tolerance(RealT tol) const -> IntT {
    return static_cast<IntT>(std::round(tol * scale));
  }

  auto deconvert_tolerance(IntT tol) const -> RealT {
    return static_cast<RealT>(tol) / scale;
  }
};

template <typename IntT, typename RealT, std::size_t Dims>
struct pt_converter
    : std::conditional_t<std::is_integral_v<RealT>,
                         pt_converter_identity<IntT, RealT, Dims>,
                         pt_converter_real<IntT, RealT, Dims>> {};

/// The whole family funnels here, so this is where an unstated `RealT` and an
/// unstated `IntT` are decided: the coordinate type first, then the lattice
/// that type calls for.
template <typename IntT = tf::none_t, typename RealT = tf::none_t,
          std::size_t Dims, typename Policy>
auto make_pt_converter(const tf::aabb_like<Dims, Policy> &aabb) {
  if constexpr (std::is_same_v<RealT, tf::none_t>) {
    return make_pt_converter<IntT, tf::coordinate_type<Policy>>(aabb);
  } else if constexpr (std::is_same_v<IntT, tf::none_t>) {
    return make_pt_converter<resolve_int_type<tf::none_t, RealT>, RealT>(aabb);
  } else if constexpr (std::is_integral_v<RealT>) {
    (void)aabb;
    return pt_converter<IntT, RealT, Dims>{};
  } else {
    auto center = aabb.center();
    auto diag = aabb.diagonal();
    RealT max_extent = 0;
    for (std::size_t i = 0; i < Dims; ++i) {
      if (diag[i] > max_extent)
        max_extent = diag[i];
    }

    constexpr auto cRangeIntMax =
        static_cast<RealT>(0.99) *
        static_cast<RealT>(std::numeric_limits<IntT>::max());
    auto scale = max_extent > 0 ? cRangeIntMax / max_extent : RealT(1);

    return pt_converter<IntT, RealT, Dims>{{scale, center}};
  }
}

template <typename IntT = tf::none_t, typename RealT = tf::none_t,
          typename Policy>
auto make_pt_converter(const tf::points<Policy> &pts) {
  auto make_aabb = [](const auto &form) {
    using P = std::decay_t<decltype(form)>;
    if constexpr (tf::has_tree_policy<P>)
      return tf::transformed(tf::aabb_from(form.tree()), tf::frame_of(form));
    else
      return tf::transformed(tf::aabb_from(form), tf::frame_of(form));
  };
  return make_pt_converter<IntT, RealT>(make_aabb(pts));
}

template <typename IntT = tf::none_t, typename RealT = tf::none_t,
          typename Policy>
auto make_pt_converter(const tf::polygons<Policy> &form) {
  auto make_aabb = [](const auto &form) {
    using P = std::decay_t<decltype(form)>;
    if constexpr (tf::has_tree_policy<P>)
      return tf::transformed(tf::aabb_from(form.tree()), tf::frame_of(form));
    else
      return tf::transformed(tf::aabb_from(form.points()), tf::frame_of(form));
  };
  return make_pt_converter<IntT, RealT>(make_aabb(form));
}

template <typename IntT = tf::none_t, typename RealT = tf::none_t,
          typename Policy>
auto make_pt_converter(const tf::segments<Policy> &form) {
  auto make_aabb = [](const auto &form) {
    using P = std::decay_t<decltype(form)>;
    if constexpr (tf::has_tree_policy<P>)
      return tf::transformed(tf::aabb_from(form.tree()), tf::frame_of(form));
    else
      return tf::transformed(tf::aabb_from(form.points()), tf::frame_of(form));
  };
  return make_pt_converter<IntT, RealT>(make_aabb(form));
}

template <typename IntT = tf::none_t, typename RealT = tf::none_t,
          typename Policy0, typename Policy1>
auto make_pt_converter(const tf::polygons<Policy0> &form0,
                       const tf::polygons<Policy1> &form1) {
  auto make_aabb = [](const auto &form) {
    using P = std::decay_t<decltype(form)>;
    if constexpr (tf::has_tree_policy<P>)
      return tf::transformed(tf::aabb_from(form.tree()), tf::frame_of(form));
    else
      return tf::transformed(tf::aabb_from(form.points()), tf::frame_of(form));
  };
  return make_pt_converter<IntT, RealT>(
      tf::aabb_union(make_aabb(form0), make_aabb(form1)));
}

template <typename IntT = tf::none_t, typename RealT = tf::none_t,
          typename Iterator, std::size_t N>
auto make_pt_converter(tf::range<Iterator, N> forms) {
  auto make_aabb = [](const auto &form) {
    using P = std::decay_t<decltype(form)>;
    if constexpr (tf::has_tree_policy<P>)
      return tf::transformed(tf::aabb_from(form.tree()), tf::frame_of(form));
    else
      return tf::transformed(tf::aabb_from(form.points()), tf::frame_of(form));
  };

  auto combined = make_aabb(forms[0]);
  for (std::size_t i = 1; i < forms.size(); ++i)
    combined = tf::aabb_union(combined, make_aabb(forms[i]));

  return make_pt_converter<IntT, RealT>(combined);
}

} // namespace tf::exact
