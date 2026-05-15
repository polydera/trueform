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

#include "trueform/geometry/compute_principal_curvatures.hpp"
#include "trueform/geometry/compute_shape_index.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"

namespace tf {
namespace ts {

template <typename Real> struct curvature_result_t {
  wasm_ndarray<Real> k0;
  wasm_ndarray<Real> k1;
};

template <typename Real> struct curvature_directions_result_t {
  wasm_ndarray<Real> k0;
  wasm_ndarray<Real> k1;
  wasm_ndarray<Real> d0;
  wasm_ndarray<Real> d1;
};

// Tag polygons with cached fm + vertex_link + point normals
template <typename Real, typename Fn>
auto with_tagged_polys(wasm_mesh<Real> &m, Fn &&fn) -> decltype(auto) {
  auto fm = m.face_membership_range();
  auto vl = m.vertex_link_range();

  // Ensure point normals are cached, create unit_vectors view
  auto pn_arr = m.point_normals();
  auto pn_view = tf::make_unit_vectors<3>(pn_arr.make_range());

  // Tag points with normals, reconstruct polygons with tagged points
  auto polys = m.polygons_range();
  auto tagged_points = polys.points() | tf::tag_normals(pn_view);
  auto tagged = tf::make_polygons(polys.faces(), tagged_points) | tf::tag(fm) |
                tf::tag(vl);

  return fn(tagged);
}

// -- Sync --

template <typename Real>
auto sync_principal_curvatures(wasm_mesh<Real> &m, int k)
    -> curvature_result_t<Real> {
  return with_tagged_polys(m, [&](auto &&polys) {
    auto [k0, k1] =
        tf::make_principal_curvatures(polys, static_cast<std::size_t>(k));
    const auto nv = m.number_of_points();
    return curvature_result_t<Real>{
        wasm_ndarray<Real>::from_buffer(std::move(k0), {nv}),
        wasm_ndarray<Real>::from_buffer(std::move(k1), {nv}),
    };
  });
}

template <typename Real>
auto sync_principal_directions(wasm_mesh<Real> &m, int k)
    -> curvature_directions_result_t<Real> {
  return with_tagged_polys(m, [&](auto &&polys) {
    auto [k0, k1, d0, d1] =
        tf::make_principal_directions(polys, static_cast<std::size_t>(k));
    const auto nv = m.number_of_points();
    return curvature_directions_result_t<Real>{
        wasm_ndarray<Real>::from_buffer(std::move(k0), {nv}),
        wasm_ndarray<Real>::from_buffer(std::move(k1), {nv}),
        wasm_ndarray<Real>::from_buffer(std::move(d0.data_buffer()), {nv, 3}),
        wasm_ndarray<Real>::from_buffer(std::move(d1.data_buffer()), {nv, 3}),
    };
  });
}

template <typename Real>
auto sync_shape_index(wasm_mesh<Real> &m, int k) -> wasm_ndarray<Real> {
  return with_tagged_polys(m, [&](auto &&polys) {
    auto si = tf::make_shape_index(polys, static_cast<std::size_t>(k));
    return wasm_ndarray<Real>::from_buffer(std::move(si),
                                           {m.number_of_points()});
  });
}

// -- Async --

template <typename Real>
auto async_principal_curvatures(wasm_mesh<Real> &m, int k) -> promise_t {
  return promise([a = m, k]() -> curvature_result_t<Real> {
    return sync_principal_curvatures<Real>(const_cast<wasm_mesh<Real> &>(a), k);
  });
}

template <typename Real>
auto async_principal_directions(wasm_mesh<Real> &m, int k) -> promise_t {
  return promise([a = m, k]() -> curvature_directions_result_t<Real> {
    return sync_principal_directions<Real>(const_cast<wasm_mesh<Real> &>(a), k);
  });
}

template <typename Real>
auto async_shape_index(wasm_mesh<Real> &m, int k) -> promise_t {
  return promise([a = m, k]() -> wasm_ndarray<Real> {
    return sync_shape_index<Real>(const_cast<wasm_mesh<Real> &>(a), k);
  });
}

} // namespace ts
} // namespace tf
