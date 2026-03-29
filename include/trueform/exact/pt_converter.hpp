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
#include <cmath>
#include <cstdint>
#include <limits>

namespace tf::exact {

template <typename RealT, std::size_t Dims> struct pt_converter {
  RealT scale;
  tf::point<RealT, Dims> offset;

  template <typename P>
  auto operator()(const P &p) const -> tf::point<int32_t, Dims> {
    tf::point<int32_t, Dims> out;
    for (std::size_t i = 0; i < Dims; ++i)
      out[i] = static_cast<int32_t>(
          std::round((static_cast<RealT>(p[i]) - offset[i]) * scale));
    return out;
  }

  auto deconvert(const tf::point<int32_t, Dims> &ip) const
      -> tf::point<RealT, Dims> {
    tf::point<RealT, Dims> out;
    for (std::size_t i = 0; i < Dims; ++i)
      out[i] = static_cast<RealT>(ip[i]) / scale + offset[i];
    return out;
  }
};

template <typename RealT = tf::none_t, std::size_t Dims, typename Policy>
auto make_pt_converter(const tf::aabb_like<Dims, Policy> &aabb) {
  if constexpr (std::is_same_v<RealT, tf::none_t>) {
    return make_pt_converter<tf::coordinate_type<Policy>>(aabb);
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
        static_cast<RealT>(std::numeric_limits<int32_t>::max());
    auto scale = max_extent > 0 ? cRangeIntMax / max_extent : RealT(1);

    return pt_converter<RealT, Dims>{scale, center};
  }
}

template <typename RealT = tf::none_t, typename Policy>
auto make_pt_converter(const tf::points<Policy> &pts) {
  auto make_aabb = [](const auto &form) {
    using P = std::decay_t<decltype(form)>;
    if constexpr (tf::has_tree_policy<P>)
      return tf::transformed(tf::aabb_from(form.tree()), tf::frame_of(form));
    else
      return tf::transformed(tf::aabb_from(form), tf::frame_of(form));
  };
  return make_pt_converter<RealT>(make_aabb(pts));
}

template <typename RealT = tf::none_t, typename Policy>
auto make_pt_converter(const tf::polygons<Policy> &form) {
  auto make_aabb = [](const auto &form) {
    using P = std::decay_t<decltype(form)>;
    if constexpr (tf::has_tree_policy<P>)
      return tf::transformed(tf::aabb_from(form.tree()), tf::frame_of(form));
    else
      return tf::transformed(tf::aabb_from(form.points()), tf::frame_of(form));
  };
  return make_pt_converter<RealT>(make_aabb(form));
}

template <typename RealT = tf::none_t, typename Policy0, typename Policy1>
auto make_pt_converter(const tf::polygons<Policy0> &form0,
                       const tf::polygons<Policy1> &form1) {
  auto make_aabb = [](const auto &form) {
    using P = std::decay_t<decltype(form)>;
    if constexpr (tf::has_tree_policy<P>)
      return tf::transformed(tf::aabb_from(form.tree()), tf::frame_of(form));
    else
      return tf::transformed(tf::aabb_from(form.points()), tf::frame_of(form));
  };
  return make_pt_converter<RealT>(
      tf::aabb_union(make_aabb(form0), make_aabb(form1)));
}

template <typename RealT = tf::none_t, typename Iterator, std::size_t N>
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

  return make_pt_converter<RealT>(combined);
}

} // namespace tf::exact
