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

#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/range.hpp"
#include "../exact/pt_converter.hpp"
#include "../exact/resolve_int_type.hpp"
#include <type_traits>

namespace tf::exact {

/// The one coordinate conversion a run shares: every operand of an
/// arrangement is converted on the same scale, so two forms name one
/// lattice.
template <typename IntT, typename RealT, std::size_t Dims>
struct vertex_converter {
  pt_converter<IntT, RealT, Dims> coords;

  /// Coordinate-only conversion (no ID).
  template <typename P>
  auto convert(const P &p) const -> tf::point<IntT, Dims> {
    return coords(p);
  }

  /// Reverse conversion (int back to float).
  auto deconvert(const tf::point<IntT, Dims> &ip) const
      -> tf::point<RealT, Dims> {
    return coords.deconvert(ip);
  }
};

template <typename IntT = tf::none_t, typename RealT = tf::none_t,
          typename Policy>
auto make_vertex_converter(const tf::polygons<Policy> &form) {
  if constexpr (std::is_same_v<RealT, tf::none_t>) {
    return make_vertex_converter<IntT, tf::coordinate_type<Policy>>(form);
  } else if constexpr (std::is_same_v<IntT, tf::none_t>) {
    return make_vertex_converter<resolve_int_type<tf::none_t, RealT>, RealT>(
        form);
  } else {
    constexpr auto Dims = tf::coordinate_dims_v<Policy>;
    auto conv = make_pt_converter<IntT, RealT>(form);
    return vertex_converter<IntT, RealT, Dims>{conv};
  }
}

template <typename IntT = tf::none_t, typename RealT = tf::none_t,
          typename Policy0, typename Policy1>
auto make_vertex_converter(const tf::polygons<Policy0> &form0,
                           const tf::polygons<Policy1> &form1) {
  if constexpr (std::is_same_v<RealT, tf::none_t>) {
    return make_vertex_converter<IntT, tf::coordinate_type<Policy0, Policy1>>(
        form0, form1);
  } else if constexpr (std::is_same_v<IntT, tf::none_t>) {
    return make_vertex_converter<resolve_int_type<tf::none_t, RealT>, RealT>(
        form0, form1);
  } else {
    constexpr auto Dims = tf::coordinate_dims_v<Policy0>;
    auto conv = make_pt_converter<IntT, RealT>(form0, form1);
    return vertex_converter<IntT, RealT, Dims>{conv};
  }
}

template <typename IntT = tf::none_t, typename RealT = tf::none_t,
          typename Iterator, std::size_t N>
auto make_vertex_converter(tf::range<Iterator, N> forms) {
  if constexpr (std::is_same_v<RealT, tf::none_t>) {
    using Policy = typename std::decay_t<decltype(forms[0])>::policy_t;
    return make_vertex_converter<IntT, tf::coordinate_type<Policy>>(forms);
  } else if constexpr (std::is_same_v<IntT, tf::none_t>) {
    return make_vertex_converter<resolve_int_type<tf::none_t, RealT>, RealT>(
        forms);
  } else {
    using form_t = std::decay_t<decltype(forms[0])>;
    constexpr auto Dims = tf::coordinate_dims_v<form_t>;
    auto conv = make_pt_converter<IntT, RealT>(forms);
    return vertex_converter<IntT, RealT, Dims>{conv};
  }
}

} // namespace tf::exact
