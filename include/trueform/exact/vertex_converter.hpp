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

#include "../core/small_vector.hpp"
#include "../exact/pt_converter.hpp"
#include "../exact/vertex.hpp"

namespace tf::exact {

/// Converts mesh-local vertices to exact vertices with globally unique IDs.
/// Wraps a pt_converter for coordinate conversion and stores per-mesh ID
/// offsets in a small_vector (stack-allocated for the common 2-mesh case).
template <typename IntT, typename RealT, std::size_t Dims>
struct vertex_converter {
  pt_converter<IntT, RealT, Dims> coords;
  tf::small_vector<int, 4> offsets;

  /// Convert a point from mesh `tag` with local index `idx`.
  template <typename Index, typename P>
  auto operator()(int tag, Index idx, const P &p) const
      -> tf::exact::vertex<Index, IntT> {
    return {idx + static_cast<Index>(offsets[tag]), coords(p)};
  }

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

template <typename IntT = int32, typename RealT = tf::none_t, typename Policy>
auto make_vertex_converter(const tf::polygons<Policy> &form) {
  if constexpr (std::is_same_v<RealT, tf::none_t>) {
    return make_vertex_converter<IntT, tf::coordinate_type<Policy>>(form);
  } else {
    constexpr auto Dims = tf::coordinate_dims_v<Policy>;
    auto conv = make_pt_converter<IntT, RealT>(form);
    return vertex_converter<IntT, RealT, Dims>{
        conv, {0, static_cast<int>(form.points().size())}};
  }
}

template <typename IntT = int32, typename RealT = tf::none_t, typename Policy0,
          typename Policy1>
auto make_vertex_converter(const tf::polygons<Policy0> &form0,
                           const tf::polygons<Policy1> &form1) {
  if constexpr (std::is_same_v<RealT, tf::none_t>) {
    return make_vertex_converter<IntT, tf::coordinate_type<Policy0, Policy1>>(
        form0, form1);
  } else {
    constexpr auto Dims = tf::coordinate_dims_v<Policy0>;
    auto conv = make_pt_converter<IntT, RealT>(form0, form1);
    return vertex_converter<IntT, RealT, Dims>{
        conv, {0, static_cast<int>(form0.points().size())}};
  }
}

template <typename IntT = int32, typename RealT = tf::none_t,
          typename Iterator, std::size_t N>
auto make_vertex_converter(tf::range<Iterator, N> forms) {
  if constexpr (std::is_same_v<RealT, tf::none_t>) {
    using Policy = typename std::decay_t<decltype(forms[0])>::policy_t;
    return make_vertex_converter<IntT, tf::coordinate_type<Policy>>(forms);
  } else {
    using form_t = std::decay_t<decltype(forms[0])>;
    constexpr auto Dims = tf::coordinate_dims_v<form_t>;
    auto conv = make_pt_converter<IntT, RealT>(forms);
    tf::small_vector<int, 4> offsets;
    int cumulative = 0;
    for (std::size_t i = 0; i < forms.size(); ++i) {
      offsets.push_back(cumulative);
      cumulative += static_cast<int>(forms[i].points().size());
    }
    return vertex_converter<IntT, RealT, Dims>{conv, std::move(offsets)};
  }
}

} // namespace tf::exact
